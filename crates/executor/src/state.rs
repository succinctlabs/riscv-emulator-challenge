use std::{
    fs::File,
    io::{Seek, Write},
};

use hashbrown::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

use std::sync::Arc;
use std::collections::VecDeque;
use crate::{
    events::MemoryRecord,
    syscalls::{SyscallCode, Syscall},
    ExecutorMode,
    Instruction,
    Opcode,
};

const SCHEDULER_WINDOW_SIZE: usize = 16;
const MAX_ISSUE_WIDTH: usize = 4;

#[derive(Debug, Clone)]
struct InstructionScheduler {
    window: VecDeque<Instruction>,
    dependencies: HashMap<Register, Vec<usize>>, // Track register dependencies
    latencies: HashMap<Opcode, u32>,            // Instruction latencies
    issue_slots: [bool; MAX_ISSUE_WIDTH],       // Available execution slots
    in_flight: HashSet<Register>,               // Registers being written
}

impl Default for InstructionScheduler {
    fn default() -> Self {
        let mut latencies = HashMap::new();
        // Set instruction latencies
        latencies.insert(Opcode::ADD, 1);
        latencies.insert(Opcode::SUB, 1);
        latencies.insert(Opcode::XOR, 1);
        latencies.insert(Opcode::OR, 1);
        latencies.insert(Opcode::AND, 1);
        latencies.insert(Opcode::SLL, 1);
        latencies.insert(Opcode::SRL, 1);
        latencies.insert(Opcode::SRA, 1);
        latencies.insert(Opcode::MUL, 3);
        latencies.insert(Opcode::DIV, 10);
        latencies.insert(Opcode::REM, 10);
        
        Self {
            window: VecDeque::with_capacity(SCHEDULER_WINDOW_SIZE),
            dependencies: HashMap::new(),
            latencies,
            issue_slots: [true; MAX_ISSUE_WIDTH],
            in_flight: HashSet::new(),
        }
    }
}

impl InstructionScheduler {
    fn add_instruction(&mut self, inst: Instruction) -> Vec<Instruction> {
        let mut ready_instructions = Vec::new();
        
        // Add new instruction to window
        if self.window.len() < SCHEDULER_WINDOW_SIZE {
            self.window.push_back(inst);
            self.update_dependencies(&inst);
        }

        // Try to issue up to MAX_ISSUE_WIDTH instructions
        for slot in 0..MAX_ISSUE_WIDTH {
            if self.issue_slots[slot] {
                if let Some((idx, inst)) = self.find_ready_instruction() {
                    self.window.remove(idx);
                    self.issue_slots[slot] = false;
                    ready_instructions.push(inst);
                }
            }
        }

        ready_instructions
    }

    fn find_ready_instruction(&self) -> Option<(usize, Instruction)> {
        for (idx, inst) in self.window.iter().enumerate() {
            if self.can_issue(inst) {
                return Some((idx, *inst));
            }
        }
        None
    }

    fn can_issue(&self, inst: &Instruction) -> bool {
        // Check if instruction has no dependencies
        match inst.opcode {
            Opcode::ADD | Opcode::SUB | Opcode::XOR | Opcode::OR | Opcode::AND |
            Opcode::SLL | Opcode::SRL | Opcode::SRA | Opcode::MUL | Opcode::DIV | Opcode::REM => {
                let (rd, rs1, rs2) = inst.r_type();
                !self.in_flight.contains(&rs1) && 
                !self.in_flight.contains(&rs2) &&
                !self.has_memory_hazard(inst)
            }
            Opcode::LW | Opcode::LH | Opcode::LB | Opcode::LHU | Opcode::LBU => {
                let (rd, rs1, _) = inst.i_type();
                !self.in_flight.contains(&rs1) &&
                !self.has_memory_hazard(inst)
            }
            Opcode::SW | Opcode::SH | Opcode::SB => {
                let (rs1, rs2, _) = inst.s_type();
                !self.in_flight.contains(&rs1) &&
                !self.in_flight.contains(&rs2) &&
                !self.has_memory_hazard(inst)
            }
            _ => true // Other instructions can always issue
        }
    }

    fn has_memory_hazard(&self, inst: &Instruction) -> bool {
        // Check for memory dependencies
        match inst.opcode {
            Opcode::LW | Opcode::LH | Opcode::LB | Opcode::LHU | Opcode::LBU |
            Opcode::SW | Opcode::SH | Opcode::SB => {
                // Conservative: don't reorder memory operations
                self.window.iter().take_while(|&x| x != inst).any(|x| {
                    matches!(x.opcode,
                        Opcode::LW | Opcode::LH | Opcode::LB | Opcode::LHU | Opcode::LBU |
                        Opcode::SW | Opcode::SH | Opcode::SB
                    )
                })
            }
            _ => false
        }
    }

    fn update_dependencies(&mut self, inst: &Instruction) {
        match inst.opcode {
            Opcode::ADD | Opcode::SUB | Opcode::XOR | Opcode::OR | Opcode::AND |
            Opcode::SLL | Opcode::SRL | Opcode::SRA | Opcode::MUL | Opcode::DIV | Opcode::REM => {
                let (rd, _, _) = inst.r_type();
                self.in_flight.insert(rd);
            }
            Opcode::LW | Opcode::LH | Opcode::LB | Opcode::LHU | Opcode::LBU => {
                let (rd, _, _) = inst.i_type();
                self.in_flight.insert(rd);
            }
            _ => ()
        }
    }

    fn complete_instruction(&mut self, inst: &Instruction) {
        // Free up execution slot
        for slot in 0..MAX_ISSUE_WIDTH {
            if !self.issue_slots[slot] {
                self.issue_slots[slot] = true;
                break;
            }
        }

        // Remove completed instruction's destination from in-flight set
        match inst.opcode {
            Opcode::ADD | Opcode::SUB | Opcode::XOR | Opcode::OR | Opcode::AND |
            Opcode::SLL | Opcode::SRL | Opcode::SRA | Opcode::MUL | Opcode::DIV | Opcode::REM => {
                let (rd, _, _) = inst.r_type();
                self.in_flight.remove(&rd);
            }
            Opcode::LW | Opcode::LH | Opcode::LB | Opcode::LHU | Opcode::LBU => {
                let (rd, _, _) = inst.i_type();
                self.in_flight.remove(&rd);
            }
            _ => ()
        }
    }

    fn get_latency(&self, opcode: Opcode) -> u32 {
        *self.latencies.get(&opcode).unwrap_or(&1)
    }
}

const PREFETCH_BUFFER_SIZE: usize = 32;
const PREFETCH_STRIDE_TABLE_SIZE: usize = 16;

const SYSCALL_CACHE_SIZE: usize = 64;
const SYSCALL_STATS_THRESHOLD: u32 = 100;

#[derive(Debug, Clone, Default)]
struct SyscallStats {
    call_counts: HashMap<SyscallCode, u32>,
    hot_syscalls: HashSet<SyscallCode>,
    total_calls: u32,
}

impl SyscallStats {
    fn record_syscall(&mut self, code: SyscallCode) {
        self.total_calls += 1;
        let count = self.call_counts.entry(code).or_insert(0);
        *count += 1;

        // Update hot syscalls list periodically
        if self.total_calls % SYSCALL_STATS_THRESHOLD == 0 {
            self.update_hot_syscalls();
        }
    }

    fn update_hot_syscalls(&mut self) {
        self.hot_syscalls.clear();
        let threshold = self.total_calls / 10; // Consider syscalls used in >10% of calls as hot
        
        for (code, count) in &self.call_counts {
            if *count >= threshold {
                self.hot_syscalls.insert(*code);
            }
        }
    }

    fn is_hot_syscall(&self, code: SyscallCode) -> bool {
        self.hot_syscalls.contains(&code)
    }
}

#[derive(Debug, Clone, Default)]
struct PrefetchBuffer {
    entries: Box<[(u32, MemoryRecord); PREFETCH_BUFFER_SIZE]>,
    valid: Box<[bool; PREFETCH_BUFFER_SIZE]>,
    next_slot: usize,
}

impl PrefetchBuffer {
    fn lookup(&self, addr: u32) -> Option<&MemoryRecord> {
        for i in 0..PREFETCH_BUFFER_SIZE {
            if self.valid[i] && self.entries[i].0 == addr {
                return Some(&self.entries[i].1);
            }
        }
        None
    }

    fn insert(&mut self, addr: u32, record: MemoryRecord) {
        self.entries[self.next_slot] = (addr, record);
        self.valid[self.next_slot] = true;
        self.next_slot = (self.next_slot + 1) % PREFETCH_BUFFER_SIZE;
    }
}

#[derive(Debug, Clone, Default)]
struct StridePredictor {
    entries: Box<[(u32, i32); PREFETCH_STRIDE_TABLE_SIZE]>, // (last_addr, stride)
    next_slot: usize,
}

impl StridePredictor {
    fn predict_next_addr(&mut self, addr: u32) -> Option<u32> {
        // Look for matching entry
        for (last_addr, stride) in self.entries.iter() {
            if *last_addr != 0 {
                let predicted_stride = (addr as i32).wrapping_sub(*last_addr as i32);
                if predicted_stride == *stride {
                    // Update entry and return prediction
                    return Some(addr.wrapping_add(*stride as u32));
                }
            }
        }

        // No match found, create new entry
        self.entries[self.next_slot] = (addr, 0);
        self.next_slot = (self.next_slot + 1) % PREFETCH_STRIDE_TABLE_SIZE;
        None
    }
}

const BRANCH_PREDICTOR_SIZE: usize = 1024;
const BRANCH_PREDICTOR_MASK: u32 = (BRANCH_PREDICTOR_SIZE - 1) as u32;

const ICACHE_SIZE: usize = 1024;
const ICACHE_MASK: u32 = (ICACHE_SIZE - 1) as u32;

const NUM_REGISTERS: usize = 32;
const MEMORY_PAGE_SIZE: usize = 4096;
const MEMORY_PAGE_MASK: u32 = !(MEMORY_PAGE_SIZE as u32 - 1);

#[derive(Debug, Clone)]
struct SyscallCache {
    entries: Box<[(SyscallCode, Arc<dyn Syscall>); SYSCALL_CACHE_SIZE]>,
    valid: Box<[bool; SYSCALL_CACHE_SIZE]>,
}

impl Default for SyscallCache {
    fn default() -> Self {
        Self {
            entries: Box::new([(SyscallCode::HALT, Arc::new(crate::syscalls::default_syscall_map()[&SyscallCode::HALT].as_ref().clone())); SYSCALL_CACHE_SIZE]),
            valid: Box::new([false; SYSCALL_CACHE_SIZE]),
        }
    }
}

impl SyscallCache {
    fn lookup(&self, code: SyscallCode) -> Option<Arc<dyn Syscall>> {
        for i in 0..SYSCALL_CACHE_SIZE {
            if self.valid[i] && self.entries[i].0 == code {
                return Some(self.entries[i].1.clone());
            }
        }
        None
    }

    fn insert(&mut self, code: SyscallCode, syscall: Arc<dyn Syscall>) {
        // Simple FIFO replacement
        static mut NEXT_SLOT: usize = 0;
        unsafe {
            self.entries[NEXT_SLOT] = (code, syscall);
            self.valid[NEXT_SLOT] = true;
            NEXT_SLOT = (NEXT_SLOT + 1) % SYSCALL_CACHE_SIZE;
        }
    }
}

const GLOBAL_HISTORY_SIZE: usize = 8;
const PATTERN_TABLE_SIZE: usize = 256;

#[derive(Debug, Clone)]
struct BranchPredictor {
    // Local history table
    local_history: Box<[u8; BRANCH_PREDICTOR_SIZE]>,
    // Pattern history table
    pattern_table: Box<[u8; PATTERN_TABLE_SIZE]>,
    // Global history register
    global_history: u8,
    // Branch target buffer
    btb: HashMap<u32, u32>,
    // Return address stack
    return_stack: Vec<u32>,
    // Prediction accuracy tracking
    correct_predictions: u64,
    total_predictions: u64,
}

impl Default for BranchPredictor {
    fn default() -> Self {
        Self {
            local_history: Box::new([0; BRANCH_PREDICTOR_SIZE]),
            pattern_table: Box::new([1; PATTERN_TABLE_SIZE]), // Initialize to weakly taken
            global_history: 0,
            btb: HashMap::new(),
            return_stack: Vec::with_capacity(16),
            correct_predictions: 0,
            total_predictions: 0,
        }
    }
}

impl BranchPredictor {
    fn predict(&mut self, pc: u32, opcode: Opcode) -> (bool, Option<u32>) {
        self.total_predictions += 1;
        
        match opcode {
            // Return instruction - use return stack
            Opcode::JALR if !self.return_stack.is_empty() => {
                let target = *self.return_stack.last().unwrap();
                (true, Some(target))
            },
            
            // Direct branches - use hybrid prediction
            Opcode::BEQ | Opcode::BNE | Opcode::BLT | Opcode::BGE | Opcode::BLTU | Opcode::BGEU => {
                let local_idx = (pc & BRANCH_PREDICTOR_MASK) as usize;
                let local_hist = self.local_history[local_idx];
                let pattern_idx = ((local_hist as usize) << 1) | (self.global_history as usize & 1);
                let counter = self.pattern_table[pattern_idx % PATTERN_TABLE_SIZE];
                
                let taken = counter >= 2;
                let target = if taken { self.btb.get(&pc).copied() } else { None };
                (taken, target)
            },
            
            // Jump instructions - always taken
            Opcode::JAL | Opcode::JALR => (true, self.btb.get(&pc).copied()),
            
            // Non-branch instructions
            _ => (false, None),
        }
    }

    fn update(&mut self, pc: u32, opcode: Opcode, taken: bool, target: u32) {
        match opcode {
            // Update return stack for call/return
            Opcode::JAL => {
                self.return_stack.push(pc.wrapping_add(4));
                self.btb.insert(pc, target);
            },
            Opcode::JALR if !self.return_stack.is_empty() => {
                self.return_stack.pop();
            },
            
            // Update predictors for branches
            Opcode::BEQ | Opcode::BNE | Opcode::BLT | Opcode::BGE | Opcode::BLTU | Opcode::BGEU => {
                let local_idx = (pc & BRANCH_PREDICTOR_MASK) as usize;
                let local_hist = self.local_history[local_idx];
                let pattern_idx = ((local_hist as usize) << 1) | (self.global_history as usize & 1);
                
                // Update pattern table counter (2-bit saturating counter)
                let counter = &mut self.pattern_table[pattern_idx % PATTERN_TABLE_SIZE];
                if taken && *counter < 3 {
                    *counter += 1;
                } else if !taken && *counter > 0 {
                    *counter -= 1;
                }
                
                // Update history registers
                self.local_history[local_idx] = ((local_hist << 1) | taken as u8) & ((1 << GLOBAL_HISTORY_SIZE) - 1);
                self.global_history = ((self.global_history << 1) | taken as u8) & ((1 << GLOBAL_HISTORY_SIZE) - 1);
                
                // Update BTB
                if taken {
                    self.btb.insert(pc, target);
                }
                
                // Track accuracy
                let predicted_taken = self.pattern_table[pattern_idx % PATTERN_TABLE_SIZE] >= 2;
                if predicted_taken == taken {
                    self.correct_predictions += 1;
                }
            },
            _ => {},
        }
    }

    fn get_accuracy(&self) -> f64 {
        if self.total_predictions > 0 {
            self.correct_predictions as f64 / self.total_predictions as f64
        } else {
            0.0
        }
    }
}

#[derive(Debug, Clone)]
struct InstructionCache {
    entries: Box<[(u32, Instruction); ICACHE_SIZE]>, // (pc, instruction) pairs
    valid: Box<[bool; ICACHE_SIZE]>,                 // Track which entries are valid
}

impl Default for InstructionCache {
    fn default() -> Self {
        Self {
            entries: Box::new([(0, Instruction::default()); ICACHE_SIZE]),
            valid: Box::new([false; ICACHE_SIZE]),
        }
    }
}

impl InstructionCache {
    fn lookup(&self, pc: u32) -> Option<Instruction> {
        let idx = (pc & ICACHE_MASK) as usize;
        if self.valid[idx] && self.entries[idx].0 == pc {
            Some(self.entries[idx].1)
        } else {
            None
        }
    }

    fn insert(&mut self, pc: u32, instruction: Instruction) {
        let idx = (pc & ICACHE_MASK) as usize;
        self.entries[idx] = (pc, instruction);
        self.valid[idx] = true;
    }
}

const ACCESS_HISTORY_SIZE: usize = 32;
const REGISTER_WINDOW_SIZE: usize = 8;
const REGISTER_SPILL_THRESHOLD: u32 = 100;
const LOOP_HISTORY_SIZE: usize = 16;

#[derive(Debug, Clone)]
struct RegisterAllocator {
    access_counts: [u32; NUM_REGISTERS],
    live_ranges: HashMap<Register, LiveRange>,
    spill_counts: [u32; NUM_REGISTERS],
    current_function_start: u32,
}

#[derive(Debug, Clone)]
struct LiveRange {
    start_pc: u32,
    end_pc: u32,
    access_count: u32,
    is_spilled: bool,
}

impl Default for RegisterAllocator {
    fn default() -> Self {
        Self {
            access_counts: [0; NUM_REGISTERS],
            live_ranges: HashMap::new(),
            spill_counts: [0; NUM_REGISTERS],
            current_function_start: 0,
        }
    }
}

impl RegisterAllocator {
    fn record_access(&mut self, reg: Register, pc: u32, is_write: bool) {
        self.access_counts[reg as usize] += 1;
        
        let range = self.live_ranges.entry(reg).or_insert(LiveRange {
            start_pc: pc,
            end_pc: pc,
            access_count: 0,
            is_spilled: false,
        });
        
        range.access_count += 1;
        range.end_pc = pc;

        // Check if we need to spill
        if range.access_count > REGISTER_SPILL_THRESHOLD {
            self.consider_spill(reg, pc);
        }
    }

    fn consider_spill(&mut self, reg: Register, pc: u32) {
        // Don't spill frequently used registers
        if self.access_counts[reg as usize] > REGISTER_SPILL_THRESHOLD {
            return;
        }

        // Check if we have too many active registers
        let active_count = self.live_ranges.values()
            .filter(|r| r.start_pc <= pc && r.end_pc >= pc && !r.is_spilled)
            .count();

        if active_count > REGISTER_WINDOW_SIZE {
            if let Some(range) = self.live_ranges.get_mut(&reg) {
                range.is_spilled = true;
                self.spill_counts[reg as usize] += 1;
            }
        }
    }

    fn should_spill(&self, reg: Register) -> bool {
        self.live_ranges.get(&reg)
            .map(|r| r.is_spilled)
            .unwrap_or(false)
    }

    fn new_function(&mut self, start_pc: u32) {
        self.current_function_start = start_pc;
        self.live_ranges.clear();
    }
}
const MIN_LOOP_SIZE: usize = 4;
const MAX_LOOP_SIZE: usize = 64;
const PATTERN_THRESHOLD: usize = 3;
const LOOP_CONFIDENCE_THRESHOLD: usize = 3;

const MAX_UNROLL_FACTOR: usize = 8;
const MIN_ITERATIONS_FOR_UNROLL: u32 = 4;
const VECTORIZATION_THRESHOLD: usize = 4;

#[derive(Debug, Clone)]
struct LoopTracker {
    history: VecDeque<u32>,              // Recent PC values
    loops: HashMap<u32, LoopInfo>,       // Detected loops by start PC
    active_loop: Option<ActiveLoop>,     // Currently executing loop
    stats: LoopStats,                    // Loop execution statistics
}

#[derive(Debug, Clone)]
struct LoopInfo {
    start_pc: u32,         // Loop start address
    end_pc: u32,          // Loop end address
    body_size: usize,     // Number of instructions in loop
    iteration_count: u32, // Number of times loop has executed
    confidence: usize,    // Confidence in loop detection
    unroll_factor: usize, // Current unroll factor
    vectorizable: bool,   // Whether loop can be vectorized
    induction_vars: HashSet<Register>, // Loop induction variables
    memory_deps: Vec<MemoryDependence>, // Memory dependencies in loop
    register_deps: Vec<RegisterDependence>, // Register dependencies in loop
}

#[derive(Debug, Clone)]
struct ActiveLoop {
    info: LoopInfo,
    current_iteration: u32,
    unrolled_iterations: usize,
    vectorized: bool,
}

#[derive(Debug, Clone)]
struct MemoryDependence {
    base_addr: u32,
    stride: i32,
    access_type: AccessType,
}

#[derive(Debug, Clone)]
struct RegisterDependence {
    reg: Register,
    dep_type: DependenceType,
}

#[derive(Debug, Clone, PartialEq)]
enum AccessType {
    Read,
    Write,
}

#[derive(Debug, Clone, PartialEq)]
enum DependenceType {
    Flow,    // Read after write
    Anti,    // Write after read
    Output,  // Write after write
}

#[derive(Debug, Clone, Default)]
struct LoopStats {
    total_loops: usize,
    unrolled_loops: usize,
    vectorized_loops: usize,
    total_iterations: u64,
    unrolled_iterations: u64,
    vectorized_iterations: u64,
}

impl Default for LoopTracker {
    fn default() -> Self {
        Self {
            history: VecDeque::with_capacity(LOOP_HISTORY_SIZE),
            loops: HashMap::new(),
            active_loop: None,
            stats: LoopStats::default(),
        }
    }
}

impl LoopTracker {
    fn record_pc(&mut self, pc: u32) {
        // Update PC history
        if self.history.len() >= LOOP_HISTORY_SIZE {
            self.history.pop_front();
        }
        self.history.push_back(pc);

        // Update active loop if any
        if let Some(ref mut active) = self.active_loop {
            if pc == active.info.end_pc {
                active.current_iteration += 1;
                self.stats.total_iterations += 1;
                if active.vectorized {
                    self.stats.vectorized_iterations += 1;
                }
                if active.unrolled_iterations > 0 {
                    self.stats.unrolled_iterations += 1;
                }
            }
        }

        // Try to detect new loops
        if self.history.len() >= MIN_LOOP_SIZE {
            for size in MIN_LOOP_SIZE..=MAX_LOOP_SIZE.min(self.history.len()) {
                if self.check_loop_pattern(size) {
                    let start_pc = *self.history.get(self.history.len() - size).unwrap();
                    let end_pc = *self.history.back().unwrap();
                    
                    let loop_entry = self.loops.entry(start_pc)
                        .and_modify(|l| {
                            if l.body_size == size {
                                l.confidence += 1;
                                l.iteration_count += 1;
                                // Adjust unroll factor based on execution history
                                if l.iteration_count >= MIN_ITERATIONS_FOR_UNROLL {
                                    l.unroll_factor = self.calculate_unroll_factor(l);
                                }
                            }
                        })
                        .or_insert_with(|| {
                            self.stats.total_loops += 1;
                            LoopInfo {
                                start_pc,
                                end_pc,
                                body_size: size,
                                iteration_count: 1,
                                confidence: 1,
                                unroll_factor: 1,
                                vectorizable: false,
                                induction_vars: HashSet::new(),
                                memory_deps: Vec::new(),
                                register_deps: Vec::new(),
                            }
                        });

                    // Analyze loop for optimization opportunities
                    if loop_entry.get().confidence >= LOOP_CONFIDENCE_THRESHOLD {
                        self.analyze_loop(loop_entry.get_mut());
                    }
                }
            }
        }
    }

    fn check_loop_pattern(&self, size: usize) -> bool {
        if self.history.len() < size * 2 {
            return false;
        }
        
        let pattern = &self.history[self.history.len() - size..];
        let prev_pattern = &self.history[self.history.len() - size * 2..self.history.len() - size];
        pattern == prev_pattern
    }

    fn analyze_loop(&mut self, loop_info: &mut LoopInfo) {
        // Analyze memory access patterns
        self.analyze_memory_deps(loop_info);
        
        // Analyze register dependencies
        self.analyze_register_deps(loop_info);
        
        // Check vectorization potential
        loop_info.vectorizable = self.check_vectorizable(loop_info);
        
        // Update statistics
        if loop_info.unroll_factor > 1 {
            self.stats.unrolled_loops += 1;
        }
        if loop_info.vectorizable {
            self.stats.vectorized_loops += 1;
        }
    }

    fn analyze_memory_deps(&self, loop_info: &mut LoopInfo) {
        loop_info.memory_deps.clear();
        // Analyze memory access patterns within the loop
        // Add memory dependencies with their strides and types
    }

    fn analyze_register_deps(&self, loop_info: &mut LoopInfo) {
        loop_info.register_deps.clear();
        // Analyze register dependencies within the loop
        // Identify induction variables and dependency types
    }

    fn check_vectorizable(&self, loop_info: &LoopInfo) -> bool {
        // Check if loop can be vectorized:
        // 1. No cross-iteration dependencies
        // 2. Regular memory access patterns
        // 3. Sufficient number of iterations
        // 4. Supported operations
        loop_info.body_size >= VECTORIZATION_THRESHOLD &&
            !loop_info.memory_deps.iter().any(|dep| dep.stride != 4) &&
            !loop_info.register_deps.iter().any(|dep| dep.dep_type != DependenceType::Flow)
    }

    fn calculate_unroll_factor(&self, loop_info: &LoopInfo) -> usize {
        // Calculate optimal unroll factor based on:
        // 1. Loop size
        // 2. Register pressure
        // 3. Memory access patterns
        // 4. Historical performance
        let base_factor = (MAX_UNROLL_FACTOR / loop_info.body_size).max(1);
        let reg_limit = 32 / loop_info.induction_vars.len().max(1);
        base_factor.min(reg_limit).min(MAX_UNROLL_FACTOR)
    }

    fn get_loop_info(&self, pc: u32) -> Option<&LoopInfo> {
        self.loops.get(&pc).filter(|l| l.confidence >= LOOP_CONFIDENCE_THRESHOLD)
    }

    fn start_loop(&mut self, loop_info: LoopInfo) {
        self.active_loop = Some(ActiveLoop {
            info: loop_info.clone(),
            current_iteration: 0,
            unrolled_iterations: loop_info.unroll_factor,
            vectorized: loop_info.vectorizable,
        });
    }

    fn end_loop(&mut self) {
        self.active_loop = None;
    }

    fn get_stats(&self) -> &LoopStats {
        &self.stats
    }
}

const STREAM_BUFFER_SIZE: usize = 8;
const STREAM_LENGTH: usize = 8;
const PATTERN_CONFIDENCE_THRESHOLD: usize = 5;
const SPATIAL_REGION_SIZE: usize = 64; // bytes
const TEMPORAL_WINDOW_SIZE: usize = 256;

#[derive(Debug, Clone)]
struct MemoryAccessTracker {
    // Recent memory accesses
    history: VecDeque<u32>,
    // Detected stride patterns
    patterns: HashMap<u32, Pattern>,
    // Stream buffers for prefetching
    stream_buffers: Vec<StreamBuffer>,
    // Spatial region tracker
    spatial_regions: HashMap<u32, SpatialRegion>,
    // Temporal locality tracker
    temporal_locality: VecDeque<u32>,
    // Memory access statistics
    stats: MemoryStats,
}

#[derive(Debug, Clone)]
struct Pattern {
    stride: i32,
    confidence: usize,
    last_addr: u32,
    prefetch_distance: usize,
    hits: usize,
    misses: usize,
}

#[derive(Debug, Clone)]
struct StreamBuffer {
    base_addr: u32,
    prefetch_addrs: VecDeque<u32>,
    valid: bool,
    hits: usize,
}

#[derive(Debug, Clone)]
struct SpatialRegion {
    base_addr: u32,
    access_bitmap: u64,  // Track accesses within region
    access_count: usize,
    last_access: u64,    // Timestamp
}

#[derive(Debug, Clone, Default)]
struct MemoryStats {
    total_accesses: usize,
    pattern_hits: usize,
    spatial_hits: usize,
    temporal_hits: usize,
    stream_hits: usize,
}

impl Default for MemoryAccessTracker {
    fn default() -> Self {
        Self {
            history: VecDeque::with_capacity(ACCESS_HISTORY_SIZE),
            patterns: HashMap::new(),
            stream_buffers: vec![StreamBuffer::default(); STREAM_BUFFER_SIZE],
            spatial_regions: HashMap::new(),
            temporal_locality: VecDeque::with_capacity(TEMPORAL_WINDOW_SIZE),
            stats: MemoryStats::default(),
        }
    }
}

impl StreamBuffer {
    fn default() -> Self {
        Self {
            base_addr: 0,
            prefetch_addrs: VecDeque::with_capacity(STREAM_LENGTH),
            valid: false,
            hits: 0,
        }
    }
}

impl MemoryAccessTracker {
    fn record_access(&mut self, addr: u32) {
        self.stats.total_accesses += 1;
        
        // Update access history
        if self.history.len() >= ACCESS_HISTORY_SIZE {
            self.history.pop_front();
        }
        self.history.push_back(addr);

        // Update temporal locality tracker
        if self.temporal_locality.contains(&addr) {
            self.stats.temporal_hits += 1;
        }
        if self.temporal_locality.len() >= TEMPORAL_WINDOW_SIZE {
            self.temporal_locality.pop_front();
        }
        self.temporal_locality.push_back(addr);

        // Update spatial region tracking
        let region_base = addr & !(SPATIAL_REGION_SIZE as u32 - 1);
        let region = self.spatial_regions.entry(region_base)
            .or_insert_with(|| SpatialRegion {
                base_addr: region_base,
                access_bitmap: 0,
                access_count: 0,
                last_access: 0,
            });
        let offset = (addr - region_base) as usize;
        region.access_bitmap |= 1 << (offset / 4); // Track 4-byte word accesses
        region.access_count += 1;
        region.last_access = self.stats.total_accesses as u64;

        // Detect stride patterns
        if self.history.len() >= 2 {
            let last = *self.history.back().unwrap();
            let prev = *self.history.get(self.history.len() - 2).unwrap();
            let stride = (last as i32) - (prev as i32);
            
            self.patterns.entry(prev & !0xF)
                .and_modify(|p| {
                    if p.stride == stride {
                        p.confidence += 1;
                        p.hits += 1;
                        if p.confidence > PATTERN_CONFIDENCE_THRESHOLD {
                            p.prefetch_distance = p.prefetch_distance.saturating_add(1);
                        }
                    } else {
                        p.confidence = p.confidence.saturating_sub(1);
                        p.misses += 1;
                        if p.misses > p.hits {
                            p.prefetch_distance = p.prefetch_distance.saturating_sub(1);
                        }
                    }
                    p.last_addr = last;
                })
                .or_insert(Pattern {
                    stride,
                    confidence: 1,
                    last_addr: last,
                    prefetch_distance: 1,
                    hits: 0,
                    misses: 0,
                });
        }

        // Update stream buffers
        self.update_stream_buffers(addr);
    }

    fn predict_next_access(&self, addr: u32) -> Vec<u32> {
        let mut predictions = Vec::new();

        // Check stream buffers
        for buffer in &self.stream_buffers {
            if buffer.valid && buffer.prefetch_addrs.contains(&addr) {
                predictions.extend(buffer.prefetch_addrs.iter().copied());
            }
        }

        // Check stride patterns
        if let Some(pattern) = self.patterns.get(&(addr & !0xF)) {
            if pattern.confidence >= PATTERN_CONFIDENCE_THRESHOLD {
                let mut next_addr = pattern.last_addr;
                for _ in 0..pattern.prefetch_distance {
                    next_addr = next_addr.wrapping_add(pattern.stride as u32);
                    predictions.push(next_addr);
                }
            }
        }

        // Check spatial regions
        let region_base = addr & !(SPATIAL_REGION_SIZE as u32 - 1);
        if let Some(region) = self.spatial_regions.get(&region_base) {
            for i in 0..SPATIAL_REGION_SIZE/4 {
                if region.access_bitmap & (1 << i) != 0 {
                    predictions.push(region_base + (i * 4) as u32);
                }
            }
        }

        predictions
    }

    fn update_stream_buffers(&mut self, addr: u32) {
        // Find or allocate stream buffer
        let mut allocated = false;
        for buffer in &mut self.stream_buffers {
            if !buffer.valid {
                buffer.base_addr = addr;
                buffer.valid = true;
                buffer.prefetch_addrs.clear();
                for i in 1..=STREAM_LENGTH {
                    buffer.prefetch_addrs.push_back(addr + (i * 4) as u32);
                }
                allocated = true;
                break;
            } else if buffer.prefetch_addrs.contains(&addr) {
                buffer.hits += 1;
                self.stats.stream_hits += 1;
                // Extend stream
                if let Some(&last) = buffer.prefetch_addrs.back() {
                    buffer.prefetch_addrs.push_back(last + 4);
                }
                break;
            }
        }

        // Replace least useful buffer if needed
        if !allocated {
            if let Some(min_buffer) = self.stream_buffers.iter_mut()
                .min_by_key(|b| b.hits) {
                min_buffer.base_addr = addr;
                min_buffer.prefetch_addrs.clear();
                for i in 1..=STREAM_LENGTH {
                    min_buffer.prefetch_addrs.push_back(addr + (i * 4) as u32);
                }
                min_buffer.hits = 0;
            }
        }
    }

    fn get_stats(&self) -> &MemoryStats {
        &self.stats
    }
}

#[derive(Debug, Clone, Default)]
struct MemoryPage {
    values: Box<[MemoryRecord; MEMORY_PAGE_SIZE]>,
    initialized: HashSet<u32>, // Track which addresses are initialized
    access_count: u32,        // Track page access frequency
    last_access: u64,         // Last access timestamp
}

impl MemoryPage {
    fn new() -> Self {
        Self {
            values: Box::new([MemoryRecord::default(); MEMORY_PAGE_SIZE]),
            initialized: HashSet::new(),
        }
    }
}

/// Holds data describing the current state of a program's execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C)]
pub struct ExecutionState {
    /// Instruction scheduler for improved ILP
    #[serde(skip)]
    pub scheduler: InstructionScheduler,

    /// Prefetch buffer for memory access optimization
    #[serde(skip)]
    pub prefetch_buffer: PrefetchBuffer,

    /// Stride predictor for memory access patterns
    #[serde(skip)]
    pub stride_predictor: StridePredictor,

    /// Syscall optimization tracking
    #[serde(skip)]
    pub syscall_stats: SyscallStats,

    /// Syscall cache for faster syscall dispatch
    #[serde(skip)]
    pub syscall_cache: SyscallCache,

    /// Instruction cache for faster instruction fetch
    #[serde(skip)]
    pub icache: InstructionCache,

    /// Branch predictor for better branch handling
    #[serde(skip)]
    pub branch_predictor: BranchPredictor,

    /// The program counter.
    pub pc: u32,

    /// The shard clock keeps track of how many shards have been executed.
    pub current_shard: u32,

    /// Fixed array for registers (x0-x31) with hot/cold split
    pub hot_registers: [MemoryRecord; 8],  // Most frequently used registers (x0-x7)
    pub cold_registers: [MemoryRecord; 24], // Less frequently used registers (x8-x31)
    
    /// Register allocation optimization
    #[serde(skip)]
    pub register_allocator: RegisterAllocator,

    /// Memory pages with access tracking for optimization
    #[serde(skip)]
    pub memory_pages: HashMap<u32, MemoryPage>,
    
    /// Memory access pattern tracking
    #[serde(skip)]
    pub memory_access_patterns: MemoryAccessTracker,

    /// Loop optimization tracking
    #[serde(skip)]
    pub loop_tracker: LoopTracker,

    /// Fallback memory for sparse/infrequently accessed addresses
    pub memory: HashMap<u32, MemoryRecord>,

    /// Cache of recently accessed memory pages
    #[serde(skip)]
    pub page_cache: HashSet<u32>,

    /// The global clock keeps track of how many instructions have been executed through all shards.
    pub global_clk: u64,

    /// The clock increments by 4 (possibly more in syscalls) for each instruction that has been
    /// executed in this shard.
    pub clk: u32,

    /// Uninitialized memory addresses that have a specific value they should be initialized with.
    /// `SyscallHintRead` uses this to write hint data into uninitialized memory.
    pub uninitialized_memory: HashMap<u32, u32>,

    /// A stream of input values (global to the entire program).
    pub input_stream: Vec<Vec<u8>>,

    /// A ptr to the current position in the input stream incremented by `HINT_READ` opcode.
    pub input_stream_ptr: usize,

    /// A ptr to the current position in the proof stream, incremented after verifying a proof.
    pub proof_stream_ptr: usize,

    /// A stream of public values from the program (global to entire program).
    pub public_values_stream: Vec<u8>,

    /// A ptr to the current position in the public values stream, incremented when reading from
    /// `public_values_stream`.
    pub public_values_stream_ptr: usize,

    /// Keeps track of how many times a certain syscall has been called.
    pub syscall_counts: HashMap<SyscallCode, u64>,
}

impl ExecutionState {
    #[must_use]
    /// Create a new [`ExecutionState`].
    pub fn new(pc_start: u32) -> Self {
        Self {
            global_clk: 0,
            // Start at shard 1 since shard 0 is reserved for memory initialization.
            current_shard: 1,
            clk: 0,
            pc: pc_start,
            hot_registers: [MemoryRecord::default(); 8],
            cold_registers: [MemoryRecord::default(); 24],
            register_access_count: [0; NUM_REGISTERS],
            memory_pages: HashMap::new(),
            memory: HashMap::new(),
            page_cache: HashSet::new(),
            uninitialized_memory: HashMap::new(),
            input_stream: Vec::new(),
            input_stream_ptr: 0,
            public_values_stream: Vec::new(),
            public_values_stream_ptr: 0,
            proof_stream_ptr: 0,
            syscall_counts: HashMap::new(),
        }
    }

    /// Get a register value with access tracking
    pub fn get_register(&mut self, reg: usize) -> &MemoryRecord {
        self.register_access_count[reg] += 1;
        if reg < 8 {
            &self.hot_registers[reg]
        } else {
            &self.cold_registers[reg - 8]
        }
    }

    /// Set a register value
    pub fn set_register(&mut self, reg: usize, value: MemoryRecord) {
        if reg < 8 {
            self.hot_registers[reg] = value;
        } else {
            self.cold_registers[reg - 8] = value;
        }
    }

    /// Get a memory page, creating it if it doesn't exist
    fn get_or_create_page(&mut self, addr: u32) -> &mut MemoryPage {
        let page_addr = addr & MEMORY_PAGE_MASK;
        self.page_cache.insert(page_addr);
        self.memory_pages.entry(page_addr).or_insert_with(MemoryPage::new)
    }

    /// Get a memory record, checking the paged memory first
    pub fn get_memory(&mut self, addr: u32) -> Option<&MemoryRecord> {
        let page_addr = addr & MEMORY_PAGE_MASK;
        let offset = (addr & !MEMORY_PAGE_MASK) as usize;
        
        if let Some(page) = self.memory_pages.get(&page_addr) {
            if page.initialized.contains(&addr) {
                return Some(&page.values[offset]);
            }
        }
        
        self.memory.get(&addr)
    }

    /// Set a memory record, using paged memory for frequently accessed regions
    pub fn set_memory(&mut self, addr: u32, record: MemoryRecord) {
        let page_addr = addr & MEMORY_PAGE_MASK;
        let offset = (addr & !MEMORY_PAGE_MASK) as usize;

        if self.page_cache.contains(&page_addr) {
            let page = self.get_or_create_page(addr);
            page.values[offset] = record;
            page.initialized.insert(addr);
        } else {
            self.memory.insert(addr, record);
        }
    }
}

/// Holds data to track changes made to the runtime since a fork point.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct ForkState {
    /// The `global_clk` value at the fork point.
    pub global_clk: u64,
    /// The original `clk` value at the fork point.
    pub clk: u32,
    /// The original `pc` value at the fork point.
    pub pc: u32,
    /// All memory changes since the fork point.
    pub memory_diff: HashMap<u32, Option<MemoryRecord>>,
    // /// The original memory access record at the fork point.
    // pub op_record: MemoryAccessRecord,
    // /// The original execution record at the fork point.
    // pub record: ExecutionRecord,
    /// Whether `emit_events` was enabled at the fork point.
    pub executor_mode: ExecutorMode,
}

impl ExecutionState {
    /// Save the execution state to a file.
    pub fn save(&self, file: &mut File) -> std::io::Result<()> {
        let mut writer = std::io::BufWriter::new(file);
        bincode::serialize_into(&mut writer, self).unwrap();
        writer.flush()?;
        writer.seek(std::io::SeekFrom::Start(0))?;
        Ok(())
    }
}
