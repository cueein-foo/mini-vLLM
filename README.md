# mini-vLLM

A simplified implementation of vLLM's core scheduling components for educational purposes.

## Features

- **Block-based Memory Management**: Efficient KV cache memory management using block-based allocation
- **FCFS Scheduling**: First Come First Served scheduling policy
- **Priority-based Preemption**: Support for request prioritization and preemption
- **Simple API**: Easy to understand and extend

## Architecture

### Scheduler (`scheduler.py`)

The main scheduling component that manages LLM inference requests:

- **BlockManager**: Manages memory blocks for KV cache allocation
- **Scheduler**: Implements scheduling logic with FCFS and preemptive scheduling
- **SchedulingRequest**: Data class representing a request with its metadata

### Key Concepts

1. **Block-based Memory**: KV cache is divided into fixed-size blocks for efficient memory management
2. **Request Lifecycle**: WAITING → RUNNING → FINISHED (or PREEMPTED → WAITING)
3. **Preemptive Scheduling**: Lower priority requests can be preempted when resources are constrained

## Usage

```python
from scheduler import Scheduler, SchedulingRequest, RequestStatus

# Create scheduler with 1000 blocks, 16 tokens per block
scheduler = Scheduler(num_blocks=1000, block_size=16, max_batch_size=32)

# Add requests
request = SchedulingRequest(
    request_id="req-1",
    prompt_length=50,
    max_tokens=100,
    priority=0
)
scheduler.add_request(request)

# Run scheduling
batch = scheduler.schedule()
print(f"Running: {[r.request_id for r in batch]}")

# Mark request as finished
scheduler.mark_finished("req-1")
```

## Running the Example

```bash
python scheduler.py
```

## Future Improvements

- [ ] Add continuous batching
- [ ] Implement block swapping to CPU memory
- [ ] Add support for prefix caching
- [ ] Implement more sophisticated scheduling policies
- [ ] Add integration with actual LLM inference

## License

MIT

## Reference

- [vLLM Paper](https://arxiv.org/abs/2309.06180)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
