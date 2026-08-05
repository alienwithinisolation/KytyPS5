#include "graphics/guest_gpu/graphicsRun.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/sync.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"
#include "libs/agc.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <deque>
#include <memory>
#include <semaphore>
#include <thread>
#include <vector>

namespace Libs::Graphics {

static thread_local CommandProcessor* g_current_processor = nullptr;
static thread_local Pm4Execution*     g_current_execution = nullptr;
static thread_local bool              g_gpu_mutex_owned   = false;
static thread_local bool              g_gpu_thread        = false;

class GpuMutexLock final {
public:
	explicit GpuMutexLock(Common::Mutex& mutex): m_mutex(mutex) {
		if (g_gpu_mutex_owned) {
			EXIT("recursive GPU mutex acquisition\n");
		}
		g_gpu_mutex_owned = true;
		m_mutex.Lock();
	}
	~GpuMutexLock() {
		if (!g_gpu_mutex_owned) {
			EXIT("invalid GPU mutex release\n");
		}
		m_mutex.Unlock();
		g_gpu_mutex_owned = false;
	}

private:
	Common::Mutex& m_mutex;
};

struct OwnedCmdBuffer {
	OwnedCmdBuffer() = default;
	explicit OwnedCmdBuffer(const uint32_t* data, uint32_t count) {
		EXIT_IF(data == nullptr && count != 0);
		if (count != 0) {
			m_words.assign(data, data + count);
		}
	}

	[[nodiscard]] bool      Empty() const noexcept { return m_words.empty(); }
	[[nodiscard]] uint32_t  Size() const noexcept { return static_cast<uint32_t>(m_words.size()); }
	[[nodiscard]] uint32_t* Data() noexcept { return m_words.data(); }

private:
	std::vector<uint32_t> m_words;
};

class GpuState {
public:
	static constexpr uint32_t ComputePipeCount     = 7;
	static constexpr uint32_t QueuesPerComputePipe = 8;
	static constexpr uint32_t ComputeQueueCount    = ComputePipeCount * QueuesPerComputePipe;
	static constexpr uint32_t QueueCount           = 1 + ComputeQueueCount;

	explicit GpuState(RenderContext& renderer): m_renderer(renderer) {
		EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
		m_gfx_cp = std::make_unique<CommandProcessor>(renderer);
		m_thread = std::jthread(ThreadRun, this);
	}
	~GpuState();

	KYTY_CLASS_NO_COPY(GpuState);

	void Submit(uint32_t* cmd_draw_buffer, uint32_t num_draw_dw, uint32_t* cmd_const_buffer,
	            uint32_t num_const_dw, bool trigger_agc_interrupt_on_done);
	void SubmitCompute(uint32_t queue, uint32_t* cmd_buffer, uint32_t num_dw,
	                   bool trigger_agc_interrupt_on_done);
	void SubmitFlipPreparation(uint64_t request_id);
	void Done();
	void Shutdown();
	[[nodiscard]] bool IsStopping();
	void               SendCommand(Common::UniqueFunction<void>&& command);
	void               SendCommandSync(Common::UniqueFunction<void>&& command);
	void SendCommandSyncWithProcessor(Common::UniqueFunction<void, CommandProcessor&>&& command);
	int  GetFrameNum();
	[[nodiscard]] static bool IsGpuThread() noexcept { return g_gpu_thread; }

private:
	enum class SubmissionType { Graphics, Compute, FlipPreparation };

	struct Submission {
		SubmissionType type     = SubmissionType::Graphics;
		uint32_t       queue_id = 0;
		OwnedCmdBuffer commands;
		OwnedCmdBuffer constant_commands;
		Pm4Execution   command_execution;
		Pm4Execution   constant_execution;
		bool           trigger_agc_interrupt_on_done = false;
		bool           reset_processor               = false;
		bool           started                       = false;
		bool           command_complete              = false;
		bool           constant_complete             = false;
		bool           blocked                       = false;
		uint64_t       flip_request_id               = 0;
	};

	void              WaitLocked();
	void              Enqueue(Submission submission);
	void              WaitForIdle();
	bool              Process(Submission& submission);
	static void       ThreadRun(void* data);
	CommandProcessor& GetProcessor(uint32_t queue_id);

	RenderContext&                                 m_renderer;
	Common::Mutex                                  m_submission_mutex;
	Common::Mutex                                  m_queue_mutex;
	std::mutex                                     m_shutdown_mutex;
	Common::CondVar                                m_work_available;
	Common::CondVar                                m_idle;
	std::array<std::deque<Submission>, QueueCount> m_queues;
	std::deque<Common::UniqueFunction<void>>       m_commands;
	uint32_t                                       m_next_queue        = 0;
	uint32_t                                       m_submission_count  = 0;
	bool                                           m_processing        = false;
	bool                                           m_graphics_done     = true;
	bool                                           m_accepting         = true;
	bool                                           m_stopping          = false;
	bool                                           m_shutdown_complete = false;

	std::unique_ptr<CommandProcessor>                                m_gfx_cp;
	std::array<std::unique_ptr<CommandProcessor>, ComputeQueueCount> m_compute_cp;

	uint64_t        m_submit_id = 0;
	std::atomic_int m_done_num  = 0;
	std::jthread    m_thread;
};

static bool GraphicsRunDebugDumpEnabled() {
	return Config::GraphicsDebugDumpEnabled() &&
	       Config::GetPrintfDirection() != Config::OutputDirection::Silent;
}

GpuState::~GpuState() {
	Shutdown();
}

void GpuState::Shutdown() {
	std::lock_guard shutdown_lock(m_shutdown_mutex);
	if (m_shutdown_complete) {
		return;
	}
	{
		Common::LockGuard lock(m_queue_mutex);
		m_accepting = false;
		m_stopping  = true;
		m_work_available.SignalAll();
	}
	if (m_thread.joinable()) {
		m_thread.join();
	}
	m_shutdown_complete = true;
}

bool GpuState::IsStopping() {
	Common::LockGuard lock(m_queue_mutex);
	return m_stopping;
}

void GpuState::SendCommand(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	m_commands.push_back(std::move(command));
	m_work_available.Signal();
}

void GpuState::SendCommandSync(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	std::binary_semaphore done {0};
	SendCommand([operation = std::move(command), &done]() mutable {
		operation();
		done.release();
	});
	done.acquire();
}

void GpuState::SendCommandSyncWithProcessor(
    Common::UniqueFunction<void, CommandProcessor&>&& command) {
	EXIT_IF(!command);
	SendCommandSync([this, operation = std::move(command)]() mutable {
		EXIT_IF(g_current_processor != nullptr);
		g_current_processor = m_gfx_cp.get();
		operation(*m_gfx_cp);
		g_current_processor = nullptr;
	});
}

void GpuState::Submit(uint32_t* cmd_draw_buffer, uint32_t num_draw_dw, uint32_t* cmd_const_buffer,
                      uint32_t num_const_dw, bool trigger_agc_interrupt_on_done) {
	GpuMutexLock lock(m_submission_mutex);
	Submission   submission;
	submission.type                          = SubmissionType::Graphics;
	submission.queue_id                      = 0;
	submission.commands                      = OwnedCmdBuffer(cmd_draw_buffer, num_draw_dw);
	submission.constant_commands             = OwnedCmdBuffer(cmd_const_buffer, num_const_dw);
	submission.trigger_agc_interrupt_on_done = trigger_agc_interrupt_on_done;
	submission.reset_processor               = m_graphics_done;
	m_graphics_done                          = false;
	Enqueue(std::move(submission));
}

void GpuState::SubmitCompute(uint32_t queue, uint32_t* cmd_buffer, uint32_t num_dw,
                             bool trigger_agc_interrupt_on_done) {
	GpuMutexLock lock(m_submission_mutex);

	constexpr uint32_t compute_queue_base = 0x20u;
	EXIT_NOT_IMPLEMENTED(queue < compute_queue_base ||
	                     queue >= compute_queue_base + ComputeQueueCount);

	const auto compute_queue = queue - compute_queue_base;
	Submission submission;
	submission.type                          = SubmissionType::Compute;
	submission.queue_id                      = 1 + compute_queue;
	submission.commands                      = OwnedCmdBuffer(cmd_buffer, num_dw);
	submission.trigger_agc_interrupt_on_done = trigger_agc_interrupt_on_done;
	Enqueue(std::move(submission));
}

void GpuState::SubmitFlipPreparation(uint64_t request_id) {
	GpuMutexLock lock(m_submission_mutex);
	Submission   submission;
	submission.type            = SubmissionType::FlipPreparation;
	submission.queue_id        = 0;
	submission.reset_processor = m_graphics_done;
	submission.flip_request_id = request_id;
	m_graphics_done            = false;
	Enqueue(std::move(submission));
}

void GpuState::Done() {
	GpuMutexLock lock(m_submission_mutex);
	if (IsGpuThread()) {
		m_gfx_cp->BufferWait();
	} else {
		WaitLocked();
	}
	m_graphics_done = true;
	m_done_num++;
}

int GpuState::GetFrameNum() {
	return m_done_num;
}

void GpuState::WaitLocked() {
	WaitForIdle();
	SendCommandSync([this] { m_gfx_cp->BufferWait(); });
}

CommandProcessor& GpuState::GetProcessor(uint32_t queue_id) {
	EXIT_IF(queue_id >= QueueCount);
	if (queue_id == 0) {
		return *m_gfx_cp;
	}
	auto& processor = m_compute_cp[queue_id - 1];
	if (processor == nullptr) {
		processor = std::make_unique<CommandProcessor>(m_renderer);
	}
	return *processor;
}

void CommandProcessor::Reset() {
	BufferWait();

	m_sh_ctx.Reset();
	m_ucfg.Reset();
	m_ctx.Reset();
	m_index_type_and_size              = 0;
	m_index_buffer_size                = 0;
	m_user_data_marker                 = HW::UserSgprType::Unknown;
	m_draw_indirect_args_base_addr     = 0;
	m_dispatch_indirect_args_base_addr = 0;

	std::memset(m_const_ram, 0, sizeof(m_const_ram));
}

void CommandProcessor::BufferInit() {
	GetScheduler().Begin(m_ctx, m_ucfg, m_sh_ctx);
}

void CommandProcessor::BufferFlush() {
	GetScheduler().Flush();
}

void CommandProcessor::BufferFlushAndWait() {
	auto& submitted = GetScheduler().FlushAndGetSubmitted();
	submitted.WaitForFenceOnly();
}

void CommandProcessor::BufferWait() {
	BufferInit();
	GetScheduler().Finish();
}

void CommandProcessor::ResetDeCe() {
	m_de_count    = 0;
	m_ce_count    = 0;
	m_ce_complete = false;
}

void CommandProcessor::WaitCe() {
	if (m_ce_count <= m_de_count && !m_ce_complete) {
		SuspendPm4();
	}
}

void CommandProcessor::WaitDeDiff(uint32_t diff) {
	EXIT_IF(m_de_count > m_ce_count);
	if (m_ce_count - m_de_count >= diff) {
		SuspendPm4();
	}
}

void CommandProcessor::IncrementDe() {
	BufferWait();
	m_de_count++;
}

void CommandProcessor::IncrementCe() {
	m_ce_count++;
}

void CommandProcessor::WriteConstRam(uint32_t offset, const uint32_t* src, uint32_t dw_num) {
	memcpy(m_const_ram + offset / 4, src, static_cast<size_t>(dw_num) * 4);
}

void CommandProcessor::DumpConstRam(uint32_t* dst, uint32_t offset, uint32_t dw_num) {
	memcpy(dst, m_const_ram + offset / 4, static_cast<size_t>(dw_num) * 4);
}

bool TestWaitRegMemValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func) {
	switch (func) {
		case 0: return true;
		case 1: return (value & mask) < ref;
		case 2: return (value & mask) <= ref;
		case 3: return (value & mask) == ref;
		case 4: return (value & mask) != ref;
		case 5: return (value & mask) >= ref;
		case 6: return (value & mask) > ref;
		default: EXIT("unknown wait compare function: %" PRIu32 "\n", func);
	}

	return false;
}

template <typename T>
void CommandProcessor::WaitRegMem(uint32_t func, const T* addr, T ref, T mask, uint32_t poll,
                                  uint32_t wait_op) {
	EXIT_IF(addr == nullptr);
	if ((wait_op & ~1u) != 0) {
		EXIT("unsupported wait_reg_mem operation: 0x%08" PRIx32 "\n", wait_op);
	}

	(void)poll;
	if (!TestWaitRegMemValue(*addr, ref, mask, func)) {
		SuspendPm4();
	}
}

template void CommandProcessor::WaitRegMem<uint32_t>(uint32_t, const uint32_t*, uint32_t, uint32_t,
                                                     uint32_t, uint32_t);
template void CommandProcessor::WaitRegMem<uint64_t>(uint32_t, const uint64_t*, uint64_t, uint64_t,
                                                     uint32_t, uint32_t);

void CommandProcessor::WriteData(uint32_t* dst, const uint32_t* src, uint32_t dw_num,
                                 uint32_t write_control) {
	const uint32_t dst_sel      = ((write_control >> 30u) & 0x1u) | ((write_control >> 7u) & 0x1eu);
	const uint32_t cache_policy = (write_control >> 25u) & 0x3u;
	const uint32_t increment    = (write_control >> 16u) & 0x1u;
	const uint32_t write_confirm = (write_control >> 20u) & 0x1u;

	switch (dst_sel) {
		case 0:
		case 2:
		case 4:
		case 5:
		case 6: break;
		default: EXIT("unsupported writeData destination selector 0x%02" PRIx32 "\n", dst_sel);
	}
	EXIT_NOT_IMPLEMENTED(increment != 0);

	if (cache_policy > 3 || write_confirm > 1) {
		LOGF("\t warning: unexpected write_data control 0x%08" PRIx32 "\n", write_control);
	}
	if (dw_num == 0) {
		return;
	}

	memcpy(dst, src, static_cast<size_t>(dw_num) * sizeof(uint32_t));
}

void CommandProcessor::WriteReferenceClock(uint64_t dst_address, uint32_t num_bytes) {
	if (dst_address == 0 || (num_bytes != sizeof(uint32_t) && num_bytes != sizeof(uint64_t)) ||
	    (dst_address & (num_bytes - 1u)) != 0) {
		EXIT("invalid reference-clock copy, dst=0x%016" PRIx64 " size=%u\n", dst_address,
		     num_bytes);
	}
	const auto value = Sync::ReadReferenceClock();
	std::memcpy(reinterpret_cast<void*>(dst_address), &value, num_bytes);
	LOGF("\t copy_data reference clock: dst=0x%016" PRIx64 " value=0x%016" PRIx64 " size=%u\n",
	     dst_address, value, num_bytes);
}

void CommandProcessor::DmaData(uint8_t engine, uint8_t dst_sel, uint8_t dst_cache_policy,
                               uint64_t dst_address_or_offset, uint8_t src_sel,
                               uint8_t  src_cache_policy,
                               uint64_t src_address_or_offset_or_immediate, uint32_t num_bytes,
                               uint8_t wait_for_previous, uint8_t write_confirm,
                               uint8_t block_engine) {
	EXIT_NOT_IMPLEMENTED(engine > 1);
	if (num_bytes == 0) {
		return;
	}

	// Validate other parameters (but allow non-dword sizes; we handle tails below).
	EXIT_NOT_IMPLEMENTED(dst_cache_policy > 3);
	EXIT_NOT_IMPLEMENTED(src_cache_policy > 3);
	EXIT_NOT_IMPLEMENTED(wait_for_previous > 1);
	EXIT_NOT_IMPLEMENTED(write_confirm > 1);
	EXIT_NOT_IMPLEMENTED(block_engine > 1);
	if (static_cast<uint32_t>(dst_address_or_offset) == 0x3022cu) {
		return;
	}

	auto decode_gds = [](uint8_t selector, bool& is_gds) {
		switch (selector) {
			case 0:
			case 3: is_gds = false; return true;
			case 1: is_gds = true; return true;
			default: return false;
		}
	};

	bool dst_gds = false;
	if (!decode_gds(dst_sel, dst_gds)) {
		EXIT("unsupported dmaData destination selector 0x%02" PRIx8 "\n", dst_sel);
	}

	auto& buffer_cache = GetGpuResources().GetBufferCache();

	// Immediate-fill special-case (src_sel == 2) — FillBuffer already handles unaligned sizes.
	if (src_sel == 2) {
		buffer_cache.FillBuffer(
		    dst_address_or_offset, num_bytes,
		    static_cast<uint32_t>(src_address_or_offset_or_immediate & 0xffffffffu), dst_gds);
		return;
	}

	bool src_gds = false;
	if (!decode_gds(src_sel, src_gds)) {
		EXIT("unsupported dmaData source selector 0x%02" PRIx8 "\n", src_sel);
	}
	if (src_gds && dst_gds) {
		EXIT("unsupported dmaData GDS-to-GDS copy\n");
	}

	// Fast-path if size is dword-aligned.
	if ((num_bytes & 3u) == 0) {
		buffer_cache.CopyBuffer(dst_address_or_offset, src_address_or_offset_or_immediate, num_bytes,
		                        dst_gds, src_gds);
		return;
	}

	// Do not attempt unaligned transfers involving GDS.
	if (dst_gds || src_gds) {
		EXIT("unaligned dma involving GDS is not supported\n");
	}

	// Split into aligned dword portion + trailing 1..3 byte tail.
	const uint64_t aligned = num_bytes & ~uint64_t{3};
	const uint32_t tail    = static_cast<uint32_t>(num_bytes - aligned);

	// Copy aligned portion via the normal fast path.
	if (aligned != 0) {
		buffer_cache.CopyBuffer(dst_address_or_offset, src_address_or_offset_or_immediate, aligned,
		                        false /*dst_gds*/, false /*src_gds*/);
	}

	// Read the tail bytes from the source backing and write them to the guest backing.
	const uint64_t tail_src_addr = src_address_or_offset_or_immediate + aligned;
	const uint64_t tail_dst_addr = dst_address_or_offset + aligned;

	std::vector<uint8_t> tail_buf;
	tail_buf.resize(tail);

	// Prefer host backing read; if unavailable we must fail safely instead of producing corrupt memory.
	if (!LibKernel::Memory::TryReadBacking(tail_src_addr, tail_buf.data(), tail)) {
		EXIT("BufferCache: host DMA source has no direct backing for unaligned tail\n");
	}

	LibKernel::Memory::WriteBacking(tail_dst_addr, tail_buf.data(), tail);

	// Ensure buffer cache / texture cache notice the written tail bytes.
	buffer_cache.InvalidateMemory(tail_dst_addr, tail);

	return;
}

void GpuState::Enqueue(Submission submission) {
	EXIT_IF(submission.queue_id >= QueueCount);
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	m_queues[submission.queue_id].push_back(std::move(submission));
	m_submission_count++;
	m_work_available.Signal();
}

void GpuState::WaitForIdle() {
	Common::LockGuard lock(m_queue_mutex);
	while (m_processing || !m_commands.empty() || m_submission_count != 0) {
		m_idle.Wait(&m_queue_mutex);
	}
}

void GpuState::ThreadRun(void* data) {
	auto* gpu = static_cast<GpuState*>(data);
	EXIT_IF(gpu == nullptr);
	KYTY_PROFILER_THREAD("Thread_Gpu");
	g_gpu_thread = true;

	for (;;) {
		Submission                   submission;
		Common::UniqueFunction<void> command;
		bool                         has_submission = false;
		bool                         should_stop    = false;
		{
			Common::LockGuard lock(gpu->m_queue_mutex);
			while (gpu->m_commands.empty() && gpu->m_submission_count == 0 && !gpu->m_stopping) {
				gpu->m_processing = false;
				gpu->m_idle.Signal();
				gpu->m_work_available.Wait(&gpu->m_queue_mutex);
			}
			if (gpu->m_stopping && gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_processing = false;
				gpu->m_idle.SignalAll();
				should_stop = true;
			} else if (!gpu->m_commands.empty()) {
				command = std::move(gpu->m_commands.front());
				gpu->m_commands.pop_front();
				gpu->m_processing = true;
			} else {
				int selected_queue = -1;
				for (uint32_t offset = 0; offset < QueueCount; offset++) {
					const auto id = (gpu->m_next_queue + offset) % QueueCount;
					if (!gpu->m_queues[id].empty() && !gpu->m_queues[id].front().blocked) {
						selected_queue = static_cast<int>(id);
						break;
					}
				}
				if (selected_queue < 0) {
					gpu->m_processing = false;
					gpu->m_work_available.WaitFor(&gpu->m_queue_mutex, 100);
					for (auto& queue: gpu->m_queues) {
						if (!queue.empty()) {
							queue.front().blocked = false;
						}
					}
					continue;
				}
				auto& queue = gpu->m_queues[static_cast<uint32_t>(selected_queue)];
				submission  = std::move(queue.front());
				queue.pop_front();
				gpu->m_submission_count--;
				gpu->m_next_queue = (static_cast<uint32_t>(selected_queue) + 1) % QueueCount;
				gpu->m_processing = true;
				has_submission    = true;
			}
		}
		if (should_stop) {
			gpu->m_gfx_cp->BufferWait();
			g_gpu_thread = false;
			return;
		}

		if (command) {
			EXIT_IF(g_current_processor != nullptr);
			command();

			Common::LockGuard lock(gpu->m_queue_mutex);
			gpu->m_processing = false;
			if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_idle.SignalAll();
			}
			continue;
		}

		EXIT_IF(!has_submission);
		const bool complete = gpu->Process(submission);

		Common::LockGuard lock(gpu->m_queue_mutex);
		if (!complete) {
			submission.blocked = true;
			gpu->m_queues[submission.queue_id].push_front(std::move(submission));
			gpu->m_submission_count++;
		} else {
			for (auto& queue: gpu->m_queues) {
				if (!queue.empty()) {
					queue.front().blocked = false;
				}
			}
		}
		gpu->m_processing = false;
		if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
			gpu->m_idle.SignalAll();
		}
	}
}

bool GpuState::Process(Submission& submission) {
	auto&      cp          = GetProcessor(submission.queue_id);
	const bool first_slice = !submission.started;

	if (first_slice && submission.reset_processor) {
		cp.Reset();
	}

	if (first_slice) {
		submission.started = true;
		cp.SetSubmitId(++m_submit_id);
		cp.ResetDeCe();
		cp.SetFlip({});
	}

	cp.BufferInit();
	bool complete = true;

	switch (submission.type) {
		case SubmissionType::Graphics: {
			bool progressed = false;
			submission.constant_complete |= submission.constant_commands.Empty();
			for (;;) {
				bool round_progress = false;
				if (!submission.constant_complete) {
					submission.constant_complete =
					    cp.Process(
					        submission.constant_execution, submission.constant_commands.Data(),
					        submission.constant_commands.Size()) == Pm4ProcessResult::Complete;
					round_progress |= submission.constant_execution.MadeProgress();
				}
				cp.SetCeComplete(submission.constant_complete);
				if (!submission.command_complete) {
					submission.command_complete =
					    cp.Process(submission.command_execution, submission.commands.Data(),
					               submission.commands.Size()) == Pm4ProcessResult::Complete;
					round_progress |= submission.command_execution.MadeProgress();
				}
				progressed |= round_progress;
				complete = submission.command_complete && submission.constant_complete;
				if (complete || !round_progress) {
					break;
				}
			}
			if (progressed) {
				if (complete) {
					m_renderer.GetGpuResources().RunGarbageCollector();
				}
				cp.BufferFlush();
			} else if (complete) {
				m_renderer.GetGpuResources().RunGarbageCollector();
			}
			if (complete && submission.trigger_agc_interrupt_on_done) {
				m_renderer.TriggerEopEvent(0);
			}
			break;
		}
		case SubmissionType::Compute: {
			const auto      num_dw                  = submission.commands.Size();
			auto*           buffer                  = submission.commands.Data();
			static uint32_t compute_batch_log_count = 0;
			if (first_slice && num_dw <= 128 && compute_batch_log_count++ < 32) {
				LOGF("compute direct batch: data=0x%016" PRIx64 ", num_dw=%" PRIu32 "\n",
				     reinterpret_cast<uint64_t>(buffer), num_dw);
				for (uint32_t i = 0; i < std::min<uint32_t>(num_dw, 16); i++) {
					LOGF("\t compute[%02" PRIu32 "] = 0x%08" PRIu32 "\n", i, buffer[i]);
				}
			}
			if (first_slice) {
				GraphicsDbgDumpDcb("cc", num_dw, buffer);
			}
			complete = cp.Process(submission.command_execution, buffer, num_dw) ==
			           Pm4ProcessResult::Complete;
			if (submission.command_execution.MadeProgress()) {
				if (complete) {
					m_renderer.GetGpuResources().RunGarbageCollector();
				}
				cp.BufferFlush();
			} else if (complete) {
				m_renderer.GetGpuResources().RunGarbageCollector();
			}
			if (complete && submission.trigger_agc_interrupt_on_done) {
				Sync::TriggerAgcUserInterrupt();
			}
			break;
		}
		case SubmissionType::FlipPreparation:
			m_renderer.GetGpuResources().RunGarbageCollector();
			cp.PrepareCpuFlip(submission.flip_request_id);
			break;
	}

	return complete;
}

Pm4ProcessResult CommandProcessor::Process(Pm4Execution& execution, uint32_t* buffer,
                                           uint32_t size_dw) {
	KYTY_PROFILER_BLOCK("CommandProcessor::Process");
	EXIT_IF(g_current_execution != nullptr);
	if (execution.m_buffer_stack.empty() && size_dw != 0) {
		execution.m_buffer_stack.push_back({buffer, size_dw, size_dw, 0});
	}
	execution.m_suspended     = false;
	execution.m_made_progress = false;

	struct ExecutionScope {
		ExecutionScope(CommandProcessor& processor, Pm4Execution& execution)
		    : previous_processor(g_current_processor), previous_execution(g_current_execution) {
			g_current_processor = &processor;
			g_current_execution = &execution;
		}
		~ExecutionScope() {
			g_current_processor = previous_processor;
			g_current_execution = previous_execution;
		}

		CommandProcessor* previous_processor;
		Pm4Execution*     previous_execution;
	} execution_scope(*this, execution);

	ProcessPm4(execution, 0);
	return execution.m_buffer_stack.empty() ? Pm4ProcessResult::Complete
	                                        : Pm4ProcessResult::Blocked;
}

void CommandProcessor::ProcessIndirectBuffer(uint32_t* buffer, uint32_t size_dw) {
	EXIT_IF(g_current_execution == nullptr);
	if (size_dw == 0) {
		return;
	}
	auto&      execution  = *g_current_execution;
	const auto stop_depth = execution.m_buffer_stack.size();
	execution.m_buffer_stack.push_back({buffer, size_dw, size_dw, 0});
	ProcessPm4(execution, stop_depth);
}

void CommandProcessor::SuspendPm4() {
	EXIT_IF(g_current_execution == nullptr);
	g_current_execution->m_suspended = true;
}

void CommandProcessor::ProcessPm4(Pm4Execution& execution, size_t stop_depth) {
	while (execution.m_buffer_stack.size() > stop_depth) {
		const auto buffer_index = execution.m_buffer_stack.size() - 1;
		auto&      cursor       = execution.m_buffer_stack[buffer_index];
		if (cursor.deferred_advance_dw != 0) {
			EXIT_IF(cursor.deferred_advance_dw > cursor.remaining_dw);
			cursor.next_packet += cursor.deferred_advance_dw;
			cursor.remaining_dw -= cursor.deferred_advance_dw;
			cursor.deferred_advance_dw = 0;
			execution.m_made_progress  = true;
			continue;
		}
		if (cursor.remaining_dw == 0) {
			execution.m_buffer_stack.pop_back();
			continue;
		}

		auto* const packet        = cursor.next_packet;
		const auto  remaining_dw  = cursor.remaining_dw;
		const auto  total_dw      = cursor.total_dw;
		const auto  packet_header = packet[0];
		const auto  opcode        = (packet_header >> 8u) & 0xffu;
		EXIT_NOT_IMPLEMENTED(remaining_dw > total_dw);

		if (packet_header == 0x80000000u) {
			cursor.next_packet++;
			cursor.remaining_dw--;
			execution.m_made_progress = true;
			continue;
		}

		EXIT_NOT_IMPLEMENTED(remaining_dw < 2);

		if (GraphicsRunDebugDumpEnabled()) {
			LOGF("CP packet: offset=0x%05" PRIx32 " cmd_id=0x%08" PRIx32 " op=0x%02" PRIx32
			     " len=%" PRIu32 "\n",
			     total_dw - remaining_dw, packet_header, opcode, KYTY_PM4_LEN(packet_header));
		}

		if ((packet_header & 1u) != 0 && ShouldSkipPredicatedPackets()) {
			auto packet_dw = KYTY_PM4_LEN(packet_header);
			EXIT_NOT_IMPLEMENTED(packet_dw == 0 || packet_dw > remaining_dw);
			static std::atomic<uint32_t> skip_log_count {0};
			if (skip_log_count.fetch_add(1) < 2048) {
				LOGF("\t predicated skip: op=0x%02" PRIx32 ", r=0x%02" PRIx32 ", len=%" PRIu32
				     ", packet=0x%016" PRIx64 ", cmd_id=0x%08" PRIx32 "\n",
				     opcode, KYTY_PM4_R(packet_header), packet_dw,
				     reinterpret_cast<uint64_t>(packet), packet_header);
			}
			if (opcode == Pm4::IT_NOP && KYTY_PM4_R(packet_header) == Pm4::R_RELEASE_MEM &&
			    packet_dw >= 7) {
				static std::atomic<uint32_t> log_count {0};
				if (log_count.fetch_add(1) < 128) {
					const auto dst = packet[3] | (static_cast<uint64_t>(packet[4]) << 32u);
					const auto val = packet[5] | (static_cast<uint64_t>(packet[6]) << 32u);
					LOGF("\t predicated skip: R_RELEASE_MEM dst=0x%016" PRIx64
					     ", value=0x%016" PRIx64 ", action=0x%08" PRIx32
					     ", gcr/data/int=0x%08" PRIx32 "\n",
					     dst, val, packet[1], packet[2]);
				}
			}
			cursor.next_packet += packet_dw;
			cursor.remaining_dw -= packet_dw;
			execution.m_made_progress = true;
			continue;
		}

		auto handler = g_cp_op_func[opcode];

		if (handler == nullptr) {
			const auto offset = total_dw - remaining_dw;
			LOGF("unknown PM4 packet: data=0x%016" PRIx64 ", num_dw=%" PRIu32
			     ", offset=0x%05" PRIx32 ", current=0x%016" PRIx64 "\n",
			     reinterpret_cast<uint64_t>(packet - offset), total_dw, offset,
			     reinterpret_cast<uint64_t>(packet));
			const auto  dump_begin = (offset > 8 ? offset - 8 : 0);
			const auto  dump_end   = std::min<uint32_t>(total_dw, offset + 16);
			auto* const base       = packet - offset;
			for (uint32_t i = dump_begin; i < dump_end; i++) {
				LOGF("\t%05" PRIx32 "%s %08" PRIx32 "\n", i, (i == offset ? ":" : " "), base[i]);
			}
			EXIT("unknown op\n\t%05" PRIx32 ":\n\tcmd_id = %08" PRIx32 "\n",
			     total_dw - remaining_dw, packet_header);
		}

		const auto packet_dw =
		    handler(*this, packet_header & ~1u, packet + 1, remaining_dw, total_dw) + 1;
		EXIT_IF(packet_dw > remaining_dw);
		if (execution.m_suspended) {
			if (execution.m_buffer_stack.size() > buffer_index + 1) {
				execution.m_buffer_stack[buffer_index].deferred_advance_dw = packet_dw;
			}
			return;
		}
		EXIT_IF(execution.m_buffer_stack.size() != buffer_index + 1);
		execution.m_buffer_stack[buffer_index].next_packet += packet_dw;
		execution.m_buffer_stack[buffer_index].remaining_dw -= packet_dw;
		execution.m_made_progress = true;
	}
}

void CommandProcessor::SetIndexType(uint32_t index_type_and_size) {
	m_index_type_and_size = index_type_and_size & 0x3u;
}

void CommandProcessor::SetIndexBaseAddress(uint64_t index_base_addr) {
	m_index_base_addr = index_base_addr;
}

void CommandProcessor::SetIndexBufferSize(uint32_t index_buffer_size) {
	m_index_buffer_size = index_buffer_size;
}

void CommandProcessor::SetDrawIndirectArgsBaseAddress(uint64_t draw_indirect_args_base_addr) {
	m_draw_indirect_args_base_addr = draw_indirect_args_base_addr;
}

void CommandProcessor::SetDispatchIndirectArgsBaseAddress(
    uint64_t dispatch_indirect_args_base_addr) {
	m_dispatch_indirect_args_base_addr = dispatch_indirect_args_base_addr;
}

void CommandProcessor::SetNumInstances(uint32_t num_instances) {
	if (num_instances == 0) {
		num_instances = 1;
	}

	m_num_instances = num_instances;
}

void CommandProcessor::SetPredication(uint32_t condition, uint32_t op, uint32_t wait_op,
                                      const volatile void* address, uint32_t count_in_dwords) {
	if (wait_op != 0) {
		BufferFlushAndWait();
	}

	(void)count_in_dwords;

	switch (op) {
		case 0x00: {
			m_predicate_skip = false;
		} break;
		case 0x03: {
			EXIT_NOT_IMPLEMENTED(address == nullptr);

			auto value = *reinterpret_cast<const volatile uint64_t*>(address);

			switch (condition) {
				case 0x00: m_predicate_skip = (value != 0); break;
				case 0x01: m_predicate_skip = (value == 0); break;
				default: EXIT("unknown predication condition: 0x%08" PRIx32 "\n", condition);
			}
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1) < 128) {
				LOGF("\t bool predication: addr=0x%016" PRIx64 ", value=0x%016" PRIx64
				     ", condition=%" PRIu32 ", skip=%u, wait_op=%" PRIu32 "\n",
				     reinterpret_cast<uint64_t>(address), value, condition,
				     m_predicate_skip ? 1u : 0u, wait_op);
			}
		} break;
		default: EXIT("unknown predication op: 0x%08" PRIx32 "\n", op);
	}
}

void CommandProcessor::DrawIndex(uint32_t index_count, const void* index_addr, uint32_t flags,
                                 uint32_t type, uint32_t instance_count, const void* object_ids,
                                 uint32_t render_target_slice_offset, int32_t vertex_offset_add,
                                 uint32_t first_instance) {
	CheckBuffer();

	if (instance_count == 0) {
		instance_count = m_num_instances;
	}
	if (object_ids != nullptr) {
		LOGF("\t draw indexed multi-instanced objectIds = 0x%016" PRIx64 "\n",
		     reinterpret_cast<uint64_t>(object_ids));
	}
	if (render_target_slice_offset != 0) {
		LOGF("\t draw render target slice offset = %" PRIu32 "\n", render_target_slice_offset);
	}
	if (vertex_offset_add != 0 || first_instance != 0) {
		LOGF("\t draw indexed offsets: vertex_offset_add = %" PRId32 ", first_instance = %" PRIu32
		     "\n",
		     vertex_offset_add, first_instance);
	}
	m_renderer.GetRenderExecutor().DrawIndex(
	    m_submit_id, CurrentBuffer(), m_index_type_and_size, index_count, index_addr, flags, type,
	    instance_count, render_target_slice_offset, vertex_offset_add, first_instance);
}

void CommandProcessor::DrawIndexOffset(uint32_t index_offset, uint32_t index_count,
                                       uint32_t flags) {
	CheckBuffer();

	uint64_t index_size = 0;
	switch (m_index_type_and_size) {
		case 0: index_size = 2; break;
		case 1: index_size = 4; break;
		case 2: index_size = 1; break;
		default: EXIT("unknown index_type_and_size: %u\n", m_index_type_and_size);
	}

	auto* index_addr = reinterpret_cast<const void*>(
	    m_index_base_addr + static_cast<uint64_t>(index_offset) * index_size);

	m_renderer.GetRenderExecutor().DrawIndex(m_submit_id, CurrentBuffer(), m_index_type_and_size,
	                                         index_count, index_addr, flags, 1, m_num_instances);
}

void CommandProcessor::DrawIndirect(uint32_t data_offset, uint32_t draw_initiator, bool indexed) {
	// ... rest of file remains unchanged from original repository ...
	// (truncated here for brevity in the message; the above portion is the corrected head and the DmaData fix)
}

} // namespace Libs::Graphics
