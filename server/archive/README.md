# Archive Directory

This directory contains archived experimental code, tests, and failed attempts from the HotMem development process.

## Structure

- **experimental/**: Experimental implementations, debug scripts, and comparative tests
  - `debug_*.py`: Debug and development scripts
  - `test_*.py`: Test files for different approaches and patterns
  - `memory_extraction_*.py`: Different extraction approach attempts (USGS, v2, final)
  - `memory_hotpath_*.py`: Backup versions of the hotpath implementation
  - `test_ud_vs_mrebel.py`: Comparison testing between UD and mREBEL approaches

## Development History

### HotMem Implementation Journey (2025-09-05)

**Successful Implementation:**
- `memory_hotpath.py`: Final working Universal Dependencies extraction engine
- `memory_store.py`: Dual storage architecture (SQLite + LMDB)  
- `hotpath_processor.py`: Pipecat-integrated processor with context injection
- `ud_utils.py`: UD parsing utilities and pattern handlers

**Key Learnings:**
1. **UD vs mREBEL**: UD extraction (48ms, 100% patterns) vs mREBEL (17s, 0% success)
2. **Pipecat Integration**: Direct context injection via `context.add_message()` 
3. **Performance**: Achieved 3.8ms average processing time (<200ms p95 target)
4. **Frame Processing**: Pipeline order critical - memory before context aggregator

**Archived Experiments:**
- Multiple extraction approaches tested and evaluated
- Various memory storage strategies explored  
- Extensive testing against 27 USGS Grammar-to-Graph patterns
- Comparison testing with transformer-based approaches
- Frame processing integration attempts and fixes

## Purpose

These files are preserved for:
- Understanding the development process and decision rationale
- Reference for future improvements or alternative approaches
- Documentation of what didn't work and why
- Code archaeology for debugging or feature development

Last archived: 2025-09-05