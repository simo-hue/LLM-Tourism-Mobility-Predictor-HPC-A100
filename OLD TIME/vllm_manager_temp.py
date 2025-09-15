# Temporary file for VLLMManager implementation

class VLLMManager:
    """Manages VLLM model for 4x A100 tensor parallel inference"""

    def __init__(self):
        self.llm: Optional[LLM] = None
        self.sampling_params: Optional[SamplingParams] = None
        self.is_initialized = False
        logger.info("🚀 VLLMManager initialized for 4x A100 tensor parallelism")

    def setup_model(self) -> bool:
        """Initialize VLLM model with 4 GPU tensor parallelism"""
        try:
            logger.info(f"🔧 Loading VLLM model: {Config.MODEL_NAME}")
            logger.info(f"🚀 Tensor Parallel Size: {Config.TENSOR_PARALLEL_SIZE} GPUs")
            logger.info(f"💾 GPU Memory Utilization: {Config.GPU_MEMORY_UTILIZATION}")

            # Initialize VLLM with 4 GPU tensor parallelism
            self.llm = LLM(
                model=Config.MODEL_NAME,
                tensor_parallel_size=Config.TENSOR_PARALLEL_SIZE,  # 4 A100 GPUs
                gpu_memory_utilization=Config.GPU_MEMORY_UTILIZATION,
                max_model_len=Config.MAX_MODEL_LEN,
                trust_remote_code=True,
                enforce_eager=False,  # Use CUDA graphs for speed
                max_num_batched_tokens=Config.BATCH_SIZE * Config.MAX_MODEL_LEN,
                max_num_seqs=Config.BATCH_SIZE,
                enable_prefix_caching=True  # Cache prefixes for speed
            )

            # Setup sampling parameters for ultra-fast generation
            self.sampling_params = SamplingParams(
                temperature=Config.TEMPERATURE,
                max_tokens=Config.MAX_TOKENS,
                top_p=Config.TOP_P,
                top_k=Config.TOP_K,
                stop=["}", "\n\n", "</s>", "<|im_end|>"],
                skip_special_tokens=True
            )

            self.is_initialized = True
            logger.info("✅ VLLM model loaded successfully on 4x A100 GPUs!")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize VLLM model: {e}")
            self.is_initialized = False
            return False

    def generate_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Generate responses for a batch of prompts - ULTRA FAST"""
        if not self.is_initialized:
            raise RuntimeError("VLLMManager not initialized. Call setup_model() first.")

        start_time = time.time()
        thread_name = threading.current_thread().name

        try:
            logger.info(f"🔍 VLLM_BATCH_START: Thread {thread_name} processing {len(prompts)} prompts")

            # Format prompts for Mistral
            formatted_prompts = []
            for prompt in prompts:
                formatted_prompt = f"<s>[INST] You are a tourism prediction assistant in Verona, Italy. {prompt} [/INST]"
                formatted_prompts.append(formatted_prompt)

            # Generate responses using VLLM
            outputs = self.llm.generate(formatted_prompts, self.sampling_params)

            # Process outputs
            results = []
            for i, output in enumerate(outputs):
                try:
                    response_text = output.outputs[0].text.strip()

                    results.append({
                        "success": True,
                        "response": {
                            "message": {
                                "content": response_text
                            }
                        },
                        "prompt_index": i,
                        "tokens_generated": len(output.outputs[0].token_ids) if output.outputs else 0
                    })

                except Exception as e:
                    logger.error(f"Error processing output {i}: {e}")
                    results.append({
                        "success": False,
                        "error": str(e),
                        "prompt_index": i
                    })

            end_time = time.time()
            duration = end_time - start_time

            logger.info(f"🔍 VLLM_BATCH_END: Thread {thread_name} completed {len(prompts)} prompts in {duration:.2f}s ({len(prompts)/duration:.1f} prompts/sec)")

            return results

        except Exception as e:
            logger.error(f"🔍 VLLM_BATCH_ERROR: Thread {thread_name} batch generation failed: {e}")
            # Return error results for all prompts
            return [{"success": False, "error": str(e), "prompt_index": i} for i in range(len(prompts))]

    def generate_single(self, prompt: str) -> Dict[str, Any]:
        """Generate response for a single prompt"""
        results = self.generate_batch([prompt])
        return results[0] if results else {"success": False, "error": "No response generated"}

    def get_chat_completion(self, prompt: str, warmup_mode: bool = False) -> Dict[str, Any]:
        """
        Compatible interface with OllamaConnectionManager for easy replacement
        """
        return self.generate_single(prompt)

    def check_models(self, expected_model: str = None) -> bool:
        """Check if model is properly loaded"""
        return self.is_initialized

    def wait_for_services(self, max_attempts: int = 30, wait_interval: int = 3) -> bool:
        """Wait for VLLM service to be ready"""
        if self.is_initialized:
            logger.info("VLLM service is ready!")
            return True
        else:
            logger.error("VLLM service is not initialized")
            return False