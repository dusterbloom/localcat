# Parallel pipelines


1. INTRO - this pipeline works until speaker recognition or straight to temporary mode MAIN 
2. MAIN  - Loop with optional memory, tools, reasoning
3. BACKUP 
4. ALT MAIN

 pipeline = Pipeline([
    transport.input(),
    ParallelPipeline(
        # Agent 1: Customer service representative
        [
            stt_1,
            context_aggregator.user_a(),
            llm_agent_1,
            tts_agent_1,
        ],
        # Agent 2: Technical specialist
        [   stt_2,
            context_aggregator.user_b(),
            llm_agent_2,
            tts_agent_2,
        ]
    ),
    transport.output(),
])