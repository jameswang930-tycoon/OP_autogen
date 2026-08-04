“hivm”方言
混合智能虚拟机（HIVM）方言。

运营
hivm.hir.atomic_cas（hivm：：AtomicCasOp）
原子比较与交换（CAS）操作

语法：

operation ::= `hivm.hir.atomic_cas` attr-dict
              `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`->` type($result_tensor)^)?
比较交换（CAS）是一种原子操作，由三个操作数组成： 内存位置（V）、期望旧值（A）、新值（B）。 操作的语义如下：V 的值更新为 B， 只有当内存位置V的值等于期望旧值A时。 无论V是否更新，操作都会返回V的原始值。

限制条件：

输入记忆法和输出记忆法必须具有相同的秩 以及相同的元素类型。

论点：

src0：期望旧值

src1：新价值

dst：GM中的内存位置

示例：

hivm.hir.atomic_cas ins(%src0, %src1 : memref<?xf32>, memref<?xf32>) outs(%dst : memref<?xf32>)
%result = hivm.hir.atomic_cas ins(%src0, %src1 : tensor<?xf32>, tensor<?xf32>) outs(%dst : tensor<?xf32>) -> tensor<?xf32>
操作数：
操作数

描述

src

张量或Memref的变异函数

dst

张量或Memref

结果：
结果

描述

result_tensor

张量或Memref

hivm.hir.atomic_rmw（hivm：：AtomicRMWOp）
原子火箭核武行动

语法：

operation ::= `hivm.hir.atomic_rmw` attr-dict
              `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `atomic_kind` `=` $atomic_kind
              (`->` type($result_tensor)^)?
原子交换是一种原子操作，包含三个步骤：

读取指定内存地址的当前值

根据 atomic_kind attr 对价值执行动作

返回之前读取的旧值 整个过程是原子的，也就是说，操作过程中不会被其他线程打断。

限制条件：

输入记忆法和输出记忆法必须具有相同的秩 以及相同的元素类型。

论点：

src：新价值

dst：GM中的内存位置

示例：

hivm.hir.atomic_rmw ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>) atomic_kind = <add>
%result = hivm.hir.atomic_rmw ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) atomic_kind = <or> -> tensor<?xf32>
Interfaces: , , , , , , DestinationStyleOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceInferCoreTypeInterfaceMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
atomic_kind	::mlir::hivm::AtomicKindAttr	
Atomic Operation Kind for StoreOp
Operands:
Operand

Description

src

any type

dst

Tensor or Memref

Results:
Result

Description

result_tensor

Tensor or Memref

hivm.hir.atomic_xchg (hivm::AtomicXchgOp)
Atomic Exchange Op

Syntax:

operation ::= `hivm.hir.atomic_xchg` attr-dict
              `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`mask` `(` $mask^ `:` type($mask) `)`)?
              (`->` type($result_tensor)^)?
Atomic exchange is an atomic operation that consists of three steps:

Read the current value of the specified memory address

Write the new value to the memory address

Return the old value read previously The whole process is atomic, that is, it will not be interrupted by other threads during the operation.

Constraints:

The input memref and output memref must have the same rank and the same element type.

Arguments:

src: new value

dst: memory location in GM

mask: mask the element

Examples:

hivm.hir.atomic_xchg ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>)
%result = hivm.hir.atomic_cas ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) -> tensor<?xf32>
Operands:
Operand

Description

src

any type

dst

Tensor or Memref

mask

Tensor or Memref

Results:
Result

Description

result_tensor

Tensor or Memref

hivm.hir.batchMmadL1 (hivm::BatchMmadL1Op)
Batch Matrix Multiply and Add Op with inputs from L1 memory hierarchy.

Syntax:

operation ::= `hivm.hir.batchMmadL1` attr-dict `ins` `(`
              $a
              `,` $b
              `,` $init_condition
              `,` $real_m
              `,` $real_k
              `,` $real_n
              (`,` $per_channel_bias^)?
              `:`
              type($a)
              `,` type($b)
              `,` type($init_condition)
              `,` type($real_m)
              `,` type($real_k)
              `,` type($real_n)
              (`,` type($per_channel_bias)^)? `)`
              `outs` `(` $c `:` type($c) `)`
              (`sync_related_args` `(` $sync_related_args^ `:` type($sync_related_args) `)`)?
              (`unit_flag` `[` $unit_flag_mode^ (`,` $unit_flag_cond^)? `]`)?
              (`->` type($result_tensors)^)?
The computation logic is:

C = C + A x B + (optional) channel_bias
Note: the rank of A, B, and C Matrix must be three, where the 0-th dimension being the batch dimension.

Traits: , , , AttrSizedOperandSegmentsCubeCoreTypeTraitMacroOpPipeTrait<PIPE::PIPE_MTE1, PIPE::PIPE_M>MacroOpTrait

Interfaces: , , , , , , , DestinationStyleOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpHIVMUnitFlagEnabledInterfaceMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
a_transpose	::mlir::UnitAttr	unit attribute
b_transpose	::mlir::UnitAttr	unit attribute
enable_HF32	::mlir::UnitAttr	unit attribute
unit_flag_mode	::mlir::hivm::UnitFlagAttr	
Operands:
Operand

Description

a

Tensor or Memref

b

Tensor or Memref

init_condition

1-bit signless integer

real_m

index

real_k

index

real_n

index

c

Tensor or Memref

sync_related_args

variadic of 64-bit signless integer

unit_flag_cond

1-bit signless integer

per_channel_bias

Tensor or Memref

Results:
Result

Description

result_tensors

variadic of ranked tensor of any type values

hivm.hir.bitcast (hivm::BitcastOp)
Reinterprets the bits of a shaped value without changing data

Syntax:

operation ::= `hivm.hir.bitcast` $src `:` type($src) `->` type($result) attr-dict
The operation converts a tensor/memref from one element type to another while preserving the underlying bit representation. The operation requires:bitcast

Same shape for input and output (2x3 != 3x2)

Same total bit-width (element_bitwidth * num_elements)

Same memory layout/strides (for memrefs)

Traits: , , AlwaysSpeculatableImplTraitElementwiseSameOperandsAndResultShape

Interfaces: , ConditionallySpeculatableNoMemoryEffect (MemoryEffectOpInterface)

Effects: MemoryEffects::Effect{}

Operands:
Operand

Description

src

any type

Results:
Result

Description

result

any type

hivm.hir.convert_layout (hivm::ConvertLayoutOp)
HIVM layout conversion operation.

Syntax:

operation ::= `hivm.hir.convert_layout` $source attr-dict `:` functional-type(operands, results)
The operation converts a memref with one layout to another. The data is not copied or modified.convert_layout

Traits: , AlwaysSpeculatableImplTraitSameOperandsAndResultElementType

Interfaces: , , , ConditionallySpeculatableInferCoreTypeInterfaceNoMemoryEffect (MemoryEffectOpInterface)ViewLikeOpInterface

Effects: MemoryEffects::Effect{}

Attributes:
Attribute	MLIR Type	Description
srcLayout	::mlir::hivm::DataLayoutAttr	
dstLayout	::mlir::hivm::DataLayoutAttr	
Operands:
Operand

Description

source

ranked or unranked memref of any type values

Results:
Result

Description

result

ranked or unranked memref of any type values

hivm.hir.copy (hivm::CopyOp)
HIVM data copy operation

Syntax:

operation ::= `hivm.hir.copy` `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              attr-dict
              (`pad_mode` `=` $pad_mode^)?
              (`pad_value` `=` $pad_value^ `:` type($pad_value))?
              (`collapse_reassociation` `=` $collapse_reassociation^)?
              (`->` type($result_tensor)^)?
Copy the data between local memory hierarchies. Currently support:

UB to UB

UB to L1 (for Ascend910_95 series)

Examples:

hivm.hir.copy ins(%src : memref<16x16xf16, #hivm.address_space<ub>>) outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
Constraints:

src and are expected to have the same element type.dst

If is not set, and shape should be the same.pad_modesrcdst

Only support left padding.

pad_value should have the same element type as and .srcdst

Non-contigous reassocicative reshape
hivm.hir.copy also supports copying non-contiguous data to contiguous storage, and vice versa. This can be seen as “expanding” or “collapsing” the data. The attribute is used to specify which axes are collapsed together. For example:collapse_reassociation

hivm.hir.copy ins(%src : memref<32x4xbf16, strided<[16, 1]>>) outs(%dst : memref<32x4xbf16, strided<[4, 1]>>)
  collapse_reassociation = [[0, 1]]
Means that the 0th and 1st axes are collapsed contiguously.

Traits: , , AlwaysSpeculatableImplTraitSinglePipeOpTraitUniformReassociationFlattenTrait

Interfaces: , , , , , , , , , ConditionallySpeculatableCopyOpInterfaceDestinationStyleOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpInferCoreTypeInterfaceMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
pad_mode	::mlir::hivm::PadModeAttr	
collapse_reassociation	::mlir::ArrayAttr	Array of 64-bit integer array attributes
Operands:
Operand

Description

src

Tensor or Memref

dst

Tensor or Memref

pad_value

any type

Results:
Result

Description

result_tensor

ranked tensor of any type values

hivm.hir.create_sync_block_lock (hivm::CreateSyncBlockLockOp)
Create sync block lock operation.

Syntax:

operation ::= `hivm.hir.create_sync_block_lock` (`from` $lockArg^)?
              attr-dict `:` (`from` type($lockArg)^ `to`)? type($memref)
The operation allocates a region of lock memory, which is used to make the code between lock and unlock execute in order among blocks. Example:create_sync_block_lock

  hivm.hir.create_sync_block_lock() : memref<1xi64>
  hivm.hir.create_sync_block_lock() from %arg : from memref<?xi8> to memref<1xi64>
Operands:
Operand

Description

lockArg

memref of any type values

Results:
Result

Description

memref

memref of any type values

hivm.hir.custom (hivm::CustomOp)
_Custom operation is a generic op interface for users to write their own custom implementation.

Scenarios:
  1. Existing operations could not fulfill the desired functionality.
  2. Existing operations could fulfill the functionality, but overall performance is not optimal.
  3. Desire for private operation._
General interface for custom op, where:

name : unique op name.

   Note : there are names reserved for builtins, usually starts with "__builtin".
          Compiler will link these builtins to self-contained template library,
          which comes together within bishengir-compile.

          For normal names/cases, user needs to specify implementation location/compilation commands (TODO),
          and all ther necessary informations.

   Available builtin names:
     "__builtin_gather_load"
inputs : input parameters.

outputs : output results, designated “init” operands, which act as initial values for the results of the operation or the init locations to which the results of the op will be written.

In order to adapt to future enhancements quickly and dynamically, custom op relies on attributes to retreive necessary information, required informations are:

CoreType : which core type to execute on, refer to TCoreTypeAttr.

Pipe : which pipe to execute on, refer to PipeAttr.

VFMode : which mode to run on vector units, refer to VFModeAttr. this attribute is ignored when core type is cube.

       Note : for builtins, user could specify these informations or not,
              compiler will help to check the correctness and canonicalize.
TODO:

Impl : user provided implementation.

Multi Pipe : custom op wants to use multiple pipes, which is a MacroOp in HIVM’s context.

Traits: , AttrSizedOperandSegmentsSinglePipeOpTrait

Interfaces: , , , , , , , , DestinationStyleOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpInferCoreTypeInterfaceMemoryEffectOpInterface (MemoryEffectOpInterface)MemoryEffectsOpInterfaceOpPipeInterface

Effects: MemoryEffects::Effect{MemoryEffects::Read on ::mlir::SideEffects::DefaultResource, MemoryEffects::Write on ::mlir::SideEffects::DefaultResource}

Attributes:
Attribute	MLIR Type	Description
name	::mlir::StringAttr	string attribute
Operands:
Operand

Description

inputs

variadic of any type

outputs

variadic of any type

Results:
Result

Description

results

variadic of any type

hivm.hir.dcci (hivm::DCCIOp)
Hivm dcci op

Syntax:

operation ::= `hivm.hir.dcci` attr-dict `(` $mode `,` $dataCacheKind (`,` $ptr^ `:` type($ptr))? `)`
This op cleans(writes back) and invalidates one cacheline or the entire data cache

Attributes:
Attribute	MLIR Type	Description
mode	::mlir::hivm::DCCIModeAttr	
hivm dcci mode
dataCacheKind	::mlir::hivm::DataCacheKindAttr	
hivm data cache kind
Operands:
Operand

Description

ptr

memref of any type values

hivm.hir.debug (hivm::DebugOp)
Device-side debugging

Syntax:

operation ::= `hivm.hir.debug` attr-dict $arg `:` type($arg)
Interfaces: , InferCoreTypeInterfaceMemoryEffectOpInterface (MemoryEffectOpInterface)

Effects: MemoryEffects::Effect{MemoryEffects::Read on ::mlir::SideEffects::DefaultResource, MemoryEffects::Write on ::mlir::SideEffects::DefaultResource}

Attributes:
Attribute	MLIR Type	Description
debugtype	::mlir::StringAttr	string attribute
prefix	::mlir::StringAttr	string attribute
hex	::mlir::BoolAttr	bool attribute
tcoretype	::mlir::hivm::TCoreTypeAttr	
Operands:
Operand

Description

arg

integer or floating-point or Tensor or Memref

hivm.hir.finish_debug (hivm::FinishDebugOp)
Finish func for device-side debugging

Syntax:

operation ::= `hivm.hir.finish_debug` attr-dict
Traits: CubeVectorCoreTypeTrait

hivm.hir.fixpipe (hivm::FixpipeOp)
HIVM data copy operation from L0C to other memory hierarchies.

Syntax:

operation ::= `hivm.hir.fixpipe` attr-dict
              `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`unit_flag` `[` $unit_flag_mode^ (`,` $unit_flag_cond^)? `]`)?
              (`->` type($result_tensor)^)?
Fixpipe is pipeline that performing data movement from L0C to other memory hierarchies, with on-the-fly fixed function of pre-stage quantization, pre-stage ReLU, element-wise add, post-stage ReLU, post-stage quantization. Currently support:

L0C to OUT

L0C to L1

L0C to UB (for Ascend910_95 series)

Additionally, Fixpipe is also capable of layout transform.

Traits: , , , AlwaysSpeculatableImplTraitCubeCoreTypeTraitOpPipeTrait<PIPE::PIPE_FIX>SinglePipeOpTrait

Interfaces: , , , , , , , , , ConditionallySpeculatableCopyOpInterfaceDestinationStyleOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpHIVMUnitFlagEnabledInterfaceMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
enable_nz2nd	::mlir::UnitAttr	unit attribute
pre_quant	::mlir::hivm::FixpipePreQuantModeAttr	
HIVM fixpipe pre_quant mode
pre_relu	::mlir::hivm::FixpipePreReluModeAttr	
HIVM fixpipe pre_relu mode
channel_split	::mlir::BoolAttr	bool attribute
unit_flag_mode	::mlir::hivm::UnitFlagAttr	
Operands:
Operand

Description

src

shaped of any type values

dst

shaped of any type values

unit_flag_cond

1-bit signless integer

Results:
Result

Description

result_tensor

ranked tensor of any type values

hivm.hir.get_block_idx (hivm::GetBlockIdxOp)
Get block idx of the current device thread used for parallelization.

Syntax:

operation ::= `hivm.hir.get_block_idx` attr-dict `->` type($result)
This op gets the block idx of the current device thread. This op will be lowered to .GetBlockIdxInstrOp

Traits: , AlwaysSpeculatableImplTraitCubeVectorCoreTypeTrait

Interfaces: , , ConditionallySpeculatableInferTypeOpInterfaceNoMemoryEffect (MemoryEffectOpInterface)

Effects: MemoryEffects::Effect{}

Results:
Result

Description

result

64-bit signless integer

hivm.hir.get_block_num (hivm::GetBlockNumOp)
Get block number of the current device thread used for parallelization.

Syntax:

operation ::= `hivm.hir.get_block_num` attr-dict `->` type($result)
This op gets the block number of the current device thread. This op will be lowered to .GetBlockNumInstrOp

Traits: , AlwaysSpeculatableImplTraitCubeVectorCoreTypeTrait

Interfaces: , , ConditionallySpeculatableInferTypeOpInterfaceNoMemoryEffect (MemoryEffectOpInterface)

Effects: MemoryEffects::Effect{}

Results:
Result

Description

result

64-bit signless integer

hivm.hir.get_sub_block_idx (hivm::GetSubBlockIdxOp)
Get sub block idx of the current device thread used for parallelization.

Syntax:

operation ::= `hivm.hir.get_sub_block_idx` attr-dict `->` type($result)
This op gets the sub block idx of the current device thread. This op will be lowered to GetSubBlockIdxInstrOp.

Traits: , AlwaysSpeculatableImplTraitCubeVectorCoreTypeTrait

Interfaces: , , ConditionallySpeculatableInferTypeOpInterfaceNoMemoryEffect (MemoryEffectOpInterface)

Effects: MemoryEffects::Effect{}

Results:
Result

Description

result

64-bit signless integer

hivm.hir.get_sub_block_num (hivm::GetSubBlockNumOp)
Get sub block number of the current device thread used for parallelization.

Syntax:

operation ::= `hivm.hir.get_sub_block_num` attr-dict `->` type($result)
This op gets the sub block number of the current device thread. This op will be lowered to GetSubBlockNumInstrOp.

Traits: , AlwaysSpeculatableImplTraitCubeVectorCoreTypeTrait

Interfaces: , , ConditionallySpeculatableInferTypeOpInterfaceNoMemoryEffect (MemoryEffectOpInterface)

Effects: MemoryEffects::Effect{}

Results:
Result

Description

result

64-bit signless integer

hivm.hir.get_sys_cnt (hivm::GetSysCntOp)
Get sys cnt of the current device

Syntax:

operation ::= `hivm.hir.get_sys_cnt` attr-dict `->` type($result)
This op get the sys cnt of the current device. This op will be lowered to .GetSysCntInstrOp

Traits: , AlwaysSpeculatableImplTraitCubeVectorCoreTypeTrait

Interfaces: , , ConditionallySpeculatableInferTypeOpInterfaceNoMemoryEffect (MemoryEffectOpInterface)

Effects: MemoryEffects::Effect{}

Results:
Result

Description

result

64-bit signless integer

hivm.hir.init_debug (hivm::InitDebugOp)
Init func for device-side debugging

Syntax:

operation ::= `hivm.hir.init_debug` attr-dict
Traits: CubeVectorCoreTypeTrait

hivm.hir.load (hivm::LoadOp)
HIVM data load operation

Syntax:

operation ::= `hivm.hir.load` `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              attr-dict
              (`pad_mode` `=` $pad_mode^)?
              (`pad_value` `=` $pad_value^ `:` type($pad_value))?
              (`left_padding_num` `=` $left_padding_num^ `:` type($left_padding_num))?
              (`init_out_buffer` `=` $init_out_buffer^ )?
              (`right_padding_num` `=` $right_padding_num^ `:` type($right_padding_num))?
              (`init_condition` `=` $init_condition^ `:` type($init_condition))?
              (`may_implicit_transpose_with_last_axis` `=` $may_implicit_transpose_with_last_axis^ )?
              (`->` type($result_tensor)^)?
Loads the data from the global memory to the local buffer. Currently only support loading to the unified buffer.

Examples:

hivm.load ins(%src : memref<16x16xf16, #hivm.address_space<gm>>) outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
Constraints:

src and are expected to have the same element type.dst

If is not set, and shape should be the same.pad_modesrcdst

Supports both left and right padding.

pad_value should have the same element type as and .srcdst

Traits: , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsOpPipeTrait<PIPE::PIPE_MTE2>SinglePipeOpTraitUniformReassociationFlattenTrait

Interfaces: , , , , , , , , , , BiShengIRAggregatedOpInterfaceConditionallySpeculatableCopyOpInterfaceDestinationStyleOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpInferCoreTypeInterfaceMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
pad_mode	::mlir::hivm::PadModeAttr	
init_out_buffer	::mlir::BoolAttr	bool attribute
may_implicit_transpose_with_last_axis	::mlir::BoolAttr	bool attribute
Operands:
Operand

Description

src

Tensor or Memref

dst

Tensor or Memref

pad_value

any type

left_padding_num

index

right_padding_num

any type

init_condition

any type

Results:
Result

Description

result_tensor

ranked tensor of any type values

hivm.hir.load_scalar (hivm::LoadScalarOp)
Hivm load scalar

Syntax:

operation ::= `hivm.hir.load_scalar` attr-dict $addr `:` type($addr) `->` type($result)
Operands:
Operand

Description

addr

LLVM pointer type

Results:
Result

Description

result

integer or floating-point

hivm.hir.matmul (hivm::MatmulOp)
HIVM Matrix Multiply Op with inputs from global memory

Syntax:

operation ::= `hivm.hir.matmul` attr-dict `ins` `(` $a `,` $b `:` type($a) `,` type($b) `)`
              `outs` `(` $c `:` type($c) `)`
              (`tiling_params` `=` $tilingParams^ `:` type($tilingParams) ) ?
              (`bias` `=` $bias^ `:` type($bias) )?
              (`descale` `=` $descale^ `:` type($descale))?
              (`a_transpose` $aTranspose^)?
              (`b_transpose` $bTranspose^)?
              (`descale_mode` `=` $descaleMode^)?
              (`block_sizes` `(` $blockSizes^ `:` type($blockSizes) `)`)?
              (`process_sizes` `(` $processSizes^ `:` type($processSizes) `)`)?
              (`swizzle_offset` `=` $swizzleOffset^ `:` type($swizzleOffset) )?
              (`swizzle_direction` `=` $swizzleDirection^ `:` type($swizzleDirection))?
              (`epilogue_p_tiles` `=` $epiloguePTiles^ `:` type($epiloguePTiles))?
              (`->` type($result)^)?
This operation takes three tiled matrices from the global memory as arguments:

A (ranked type): an matrixm x k

B (ranked type): an matrixk x n

C (ranked type): an matrixm x n

Other arguments include:

block_sizes: data size of m, n, and k dimension processed on the L1 memory hierarchy

process_sizes: data size of m, n, and k dimension processed on the L0 memory hierarchy

(optional) : continuous block number which swizzle scheduleswizzle_offset

(optional) : block direction which swizzle scheduleswizzle_direction

(optional) : block number which compute attached op once handleepilogue_p_tiles

The operation performed is represented as . If or is present, the respective operand is loaded in a transposed manner.C = A * Ba_transposeb_transpose

Optionally, this operation takes the following arguments:

bias (ranked type): bias value, which is a vector of shape n

descale: antiquantionzation value. Support 3 types:

DescaleNull : no descale.

DescalePerChannel: the shape of is equal to .descalen

DescalePerTensor: the shape of is equal to .descale1

The operation performed is represented as .C = descale * (A * B + bias)

Traits: , , AttrSizedOperandSegmentsMacroOpPipeTrait<PIPE::PIPE_MTE2, PIPE::PIPE_MTE3>MacroOpTrait

Interfaces: , , , , , , , DestinationStyleOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpInferCoreTypeInterfaceMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
aTranspose	::mlir::UnitAttr	unit attribute
bTranspose	::mlir::UnitAttr	unit attribute
descaleMode	::mlir::hivm::DescaleModeAttr	
descale mode for matmul
Operands:
Operand

Description

a

shaped of any type values

b

shaped of any type values

tilingParams

shaped of any type values

bias

shaped of any type values

descale

shaped of any type values

blockSizes

variadic of 64-bit signless integer

processSizes

variadic of 64-bit signless integer

swizzleOffset

64-bit signless integer

swizzleDirection

64-bit signless integer

epiloguePTiles

64-bit signless integer

c

shaped of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.mix_group_matmul (hivm::MixGroupMatmulOp)
HIVM (Mix) Matrix Group Multiply Op with inputs from global memory

Syntax:

operation ::= `hivm.hir.mix_group_matmul` attr-dict `ins` `(` $a `,` $b `,` $tokens_per_expert `:` type($a) `,` type($b) `,` type($tokens_per_expert) `)`
              (`post_vector_func_ins` `(` $postVecFuncIns^ `:` type($postVecFuncIns) `)`) ?
              (`post_vector_func_outs` `(` $postVecFuncOuts^ `:` type($postVecFuncOuts) `)`) ?
              (`workspace_ins` `(` $workspaceIns^ `:` type($workspaceIns) `)`) ?
              `outs` `(` $c `:` type($c) `)`
              (`tiling_params` `=` $tilingParams^ `:` type($tilingParams) ) ?
              (`comm_params` `=` $commParams^ `:` type($commParams) ) ?
              (`bias` `=` $bias^ `:` type($bias) )?
              (`descale` `=` $descale^ `:` type($descale))?
              (`a_transpose` $aTranspose^)?
              (`b_transpose` $bTranspose^)?
              (`descale_mode` `=` $descaleMode^)?
              (`block_sizes` `(` $blockSizes^ `:` type($blockSizes) `)`)?
              (`process_sizes` `(` $processSizes^ `:` type($processSizes) `)`)?
              (`swizzle_offset` `=` $swizzleOffset^ `:` type($swizzleOffset) )?
              (`swizzle_direction` `=` $swizzleDirection^ `:` type($swizzleDirection))?
              (`epilogue_p_tiles` `=` $epiloguePTiles^ `:` type($epiloguePTiles))?
              (`->` type($result)^)?
This operation takes three tiled matrices from the global memory as arguments:

A (ranked type): an matrixm x k

B (ranked type): an matrixk x n

C (ranked type): an matrixm x n

Other arguments include:

block_sizes: data size of m, n, and k dimension processed on the L1 memory hierarchy

process_sizes: data size of m, n, and k dimension processed on the L0 memory hierarchy

(optional) : continuous block number which swizzle scheduleswizzle_offset

(optional) : block direction which swizzle scheduleswizzle_direction

(optional) : block number which compute attached op once handleepilogue_p_tiles

The operation performed is represented as . If or is present, the respective operand is loaded in a transposed manner.C = A * Ba_transposeb_transpose

Optionally, this operation takes the following arguments:

bias (ranked type): bias value, which is a vector of shape n

descale: antiquantionzation value. Support 3 types:

DescaleNull : no descale.

DescalePerChannel: the shape of is equal to .descalen

DescalePerTensor: the shape of is equal to .descale1

The operation performed is represented as .C = descale * (A * B + bias)

This operation also supports tile-level fusion with a post-vector function (hence it’s a Mix op) specify how matmuls are distrubuted to different experts is used to specify the arguments. is used to specify the outputs. is used to specify communication related arguments (eg. topology, communicator, group, etc al) when fusing communication operators.tokens_per_expertpost_vector_func_inspost_vector_func_outscomm_params

Traits: , , AttrSizedOperandSegmentsMacroOpPipeTrait<PIPE::PIPE_MTE2, PIPE::PIPE_MTE3>MacroOpTrait

Interfaces: , , , , , , , DestinationStyleOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpInferCoreTypeInterfaceMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
aTranspose	::mlir::UnitAttr	unit attribute
bTranspose	::mlir::UnitAttr	unit attribute
descaleMode	::mlir::hivm::DescaleModeAttr	
descale mode for matmul
Operands:
Operand

Description

a

shaped of any type values

b

shaped of any type values

tokens_per_expert

shaped of any type values

postVecFuncIns

variadic of shaped of any type values

postVecFuncOuts

variadic of shaped of any type values

workspaceIns

variadic of shaped of any type values

tilingParams

shaped of any type values

commParams

shaped of any type values

bias

shaped of any type values

descale

shaped of any type values

blockSizes

variadic of 64-bit signless integer

processSizes

variadic of 64-bit signless integer

swizzleOffset

64-bit signless integer

swizzleDirection

64-bit signless integer

epiloguePTiles

64-bit signless integer

c

shaped of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.mix_matmul (hivm::MixMatmulOp)
HIVM (Mix) Matrix Multiply Op with inputs from global memory

Syntax:

operation ::= `hivm.hir.mix_matmul` attr-dict `ins` `(` $a `,` $b `:` type($a) `,` type($b) `)`
              (`post_vector_func_ins` `(` $postVecFuncIns^ `:` type($postVecFuncIns) `)`) ?
              (`workspace_ins` `(` $workspaceIns^ `:` type($workspaceIns) `)`) ?
              `outs` `(` $c `:` type($c) `)`
              (`tiling_params` `=` $tilingParams^ `:` type($tilingParams) ) ?
              (`comm_params` `=` $commParams^ `:` type($commParams) ) ?
              (`bias` `=` $bias^ `:` type($bias) )?
              (`descale` `=` $descale^ `:` type($descale))?
              (`a_transpose` $aTranspose^)?
              (`b_transpose` $bTranspose^)?
              (`descale_mode` `=` $descaleMode^)?
              (`block_sizes` `(` $blockSizes^ `:` type($blockSizes) `)`)?
              (`process_sizes` `(` $processSizes^ `:` type($processSizes) `)`)?
              (`swizzle_offset` `=` $swizzleOffset^ `:` type($swizzleOffset) )?
              (`swizzle_direction` `=` $swizzleDirection^ `:` type($swizzleDirection))?
              (`epilogue_p_tiles` `=` $epiloguePTiles^ `:` type($epiloguePTiles))?
              (`->` type($result)^)?
This operation takes three tiled matrices from the global memory as arguments:

A (ranked type): an matrixm x k

B (ranked type): an matrixk x n

C (ranked type): an matrixm x n

Other arguments include:

block_sizes: data size of m, n, and k dimension processed on the L1 memory hierarchy

process_sizes: data size of m, n, and k dimension processed on the L0 memory hierarchy

(optional) : continuous block number which swizzle scheduleswizzle_offset

(optional) : block direction which swizzle scheduleswizzle_direction

(optional) : block number which compute attached op once handleepilogue_p_tiles

The operation performed is represented as . If or is present, the respective operand is loaded in a transposed manner.C = A * Ba_transposeb_transpose

Optionally, this operation takes the following arguments:

bias (ranked type): bias value, which is a vector of shape n

descale: antiquantionzation value. Support 3 types:

DescaleNull : no descale.

DescalePerChannel: the shape of is equal to .descalen

DescalePerTensor: the shape of is equal to .descale1

The operation performed is represented as .C = descale * (A * B + bias)

This operation also supports tile-level fusion with a post-vector function (hence it’s a Mix op). is used to specify the arguments. is used to specify communication related arguments (eg. topology, communicator, group, etc al) when fusing communication operators.post_vector_func_inscomm_params

Traits: , , AttrSizedOperandSegmentsMacroOpPipeTrait<PIPE::PIPE_MTE2, PIPE::PIPE_MTE3>MacroOpTrait

Interfaces: , , , , , , , DestinationStyleOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpInferCoreTypeInterfaceMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
aTranspose	::mlir::UnitAttr	unit attribute
bTranspose	::mlir::UnitAttr	unit attribute
descaleMode	::mlir::hivm::DescaleModeAttr	
descale mode for matmul
Operands:
Operand

Description

a

shaped of any type values

b

shaped of any type values

postVecFuncIns

variadic of shaped of any type values

workspaceIns

variadic of shaped of any type values

tilingParams

shaped of any type values

commParams

shaped of any type values

bias

shaped of any type values

descale

shaped of any type values

blockSizes

variadic of 64-bit signless integer

processSizes

variadic of 64-bit signless integer

swizzleOffset

64-bit signless integer

swizzleDirection

64-bit signless integer

epiloguePTiles

64-bit signless integer

c

shaped of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.mmadL1 (hivm::MmadL1Op)
Matrix Multiply and Add Op with inputs from L1 memory hierarchy.

Syntax:

operation ::= `hivm.hir.mmadL1` attr-dict `ins` `(`
              $a
              `,` $b
              `,` $init_condition
              `,` $real_m
              `,` $real_k
              `,` $real_n
              (`,` $per_channel_bias^)?
              `:`
              type($a)
              `,` type($b)
              `,` type($init_condition)
              `,` type($real_m)
              `,` type($real_k)
              `,` type($real_n)
              (`,` type($per_channel_bias)^)? `)`
              `outs` `(` $c `:` type($c) `)`
              (`sync_related_args` `(` $sync_related_args^ `:` type($sync_related_args) `)`)?
              (`unit_flag` `[` $unit_flag_mode^ (`,` $unit_flag_cond^)? `]`)?
              (`->` type($result_tensors)^)?
The computation logic is:

C = C + A x B + (optional) channel_bias
Note: the rank of A, B, and C Matrix must be two.

Traits: , , , AttrSizedOperandSegmentsCubeCoreTypeTraitMacroOpPipeTrait<PIPE::PIPE_MTE1, PIPE::PIPE_M>MacroOpTrait

Interfaces: , , , , , , , , DestinationStyleOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpHIVMUnitFlagEnabledInterfaceMemoryEffectsOpInterfaceOpLayoutInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
a_transpose	::mlir::UnitAttr	unit attribute
b_transpose	::mlir::UnitAttr	unit attribute
enable_HF32	::mlir::UnitAttr	unit attribute
unit_flag_mode	::mlir::hivm::UnitFlagAttr	
Operands:
Operand

Description

a

Tensor or Memref

b

Tensor or Memref

init_condition

1-bit signless integer

real_m

index

real_k

index

real_n

index

c

Tensor or Memref

sync_related_args

variadic of 64-bit signless integer

unit_flag_cond

1-bit signless integer

per_channel_bias

Tensor or Memref

Results:
Result

Description

result_tensors

variadic of ranked tensor of any type values

hivm.hir.nd2nz (hivm::ND2NZOp)
HIVM data copy operation with on-the-fly ND to NZ layout transformation

Syntax:

operation ::= `hivm.hir.nd2nz` attr-dict
              `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`init_out_buffer` `=` $init_out_buffer^ )?
              (`pad_value` `=` $pad_value^ `:` type($pad_value))?
              (`init_condition` `=` $init_condition^ `:` type($init_condition))?
              (`->` type($result_tensor)^)?
dst_continuous: if present, signify that the source data is stored continuously in the destination buffer. This must be set in order for this op to be converted to library function call. Constraints:

if is true, should have value.init_out_bufferpad_value

Traits: , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsCubeCoreTypeTraitOpPipeTrait<PIPE::PIPE_MTE2>SinglePipeOpTrait

Interfaces: , , , , , , , , , BiShengIRAggregatedOpInterfaceConditionallySpeculatableCopyOpInterfaceDestinationStyleOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
dst_continuous	::mlir::UnitAttr	unit attribute
init_out_buffer	::mlir::BoolAttr	bool attribute
Operands:
Operand

Description

src

shaped of any type values

dst

shaped of any type values

pad_value

any type

init_condition

any type

Results:
Result

Description

result_tensor

variadic of ranked tensor of any type values

hivm.hir.nz2nd (hivm::NZ2NDOp)
HIVM data copy operation from L1 to Global Memory with NZ2ND conversion

Syntax:

operation ::= `hivm.hir.nz2nd` attr-dict
              `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`->` type($result_tensor)^)?
NZ2ND does data movement from L1 to OUT with NZ2ND conversion. Traits: , , , AlwaysSpeculatableImplTraitCubeCoreTypeTraitOpPipeTrait<PIPE::PIPE_MTE3>SinglePipeOpTrait

Interfaces: , , , , , , , , ConditionallySpeculatableCopyOpInterfaceDestinationStyleOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpMemoryEffectsOpInterfaceOpPipeInterface

Operands:
Operand

Description

src

Tensor or Memref

dst

Tensor or Memref

Results:
Result

Description

result_tensor

ranked tensor of any type values

hivm.hir.pipe_barrier (hivm::PipeBarrierOp)
Hivm pipe barrier.

Syntax:

operation ::= `hivm.hir.pipe_barrier` `[` $pipe `]` attr-dict
Interfaces: InferCoreTypeInterface

Attributes:
Attribute	MLIR Type	Description
pipe	::mlir::hivm::PipeAttr	
hivm.hir.pointer_cast (hivm::PointerCastOp)
HIVM pointer cast op at specific i64 addr

Syntax:

operation ::= `hivm.hir.pointer_cast` `(`$addrs `)` (`[` $dynamicSizes^`]`)? attr-dict `:` type($result)
The specific i64 addrs are stored in , which is variadic.$addrs

Constraints:

The type of each address should be i64.

addrs should have at least one addr.

Examples:

%addr = arith.constant 1234 : i64
%tmp = hivm.hir.pointer_cast(%addr) : memref<32xf32>

%addr2 = arith.constant 1600 : i64
%addr3 = arith.constant 3200 : i64
%tmp2 = hivm.hir.pointer_cast(%addr, %addr2) : memref<32xf32>
%tmp3 = hivm.hir.pointer_cast(%addr, %addr2, %addr3) : memref<32xf32>
Traits: , AttrSizedOperandSegmentsCubeVectorCoreTypeTrait

Operands:
Operand

Description

addrs

variadic of 64-bit signless integer

dynamicSizes

variadic of index

Results:
Result

Description

result

memref of any type values

hivm.hir.set_ffts_base_addr (hivm::SetFFTSBaseAddrOp)
Set base addr for ffts sync machenism.

Syntax:

operation ::= `hivm.hir.set_ffts_base_addr` attr-dict $ffts_base_addr
Traits: CubeVectorCoreTypeTrait

Operands:
Operand

Description

ffts_base_addr

64-bit signless integer

hivm.hir.set_flag (hivm::SetFlagOp)
Hivm set flag.

Syntax:

operation ::= `hivm.hir.set_flag` `[`
              $set_pipe
              `,` $wait_pipe
              `,` custom<EventID>($static_event_id, $dynamic_event_id)
              `]` attr-dict
Interfaces: InferCoreTypeInterface

Attributes:
Attribute	MLIR Type	Description
set_pipe	::mlir::hivm::PipeAttr	
wait_pipe	::mlir::hivm::PipeAttr	
static_event_id	::mlir::hivm::EventAttr	
Operands:
Operand

Description

dynamic_event_id

64-bit signless integer

hivm.hir.set_mask_norm (hivm::SetMaskNormOp)
Hivm set mask norm

Syntax:

operation ::= `hivm.hir.set_mask_norm` attr-dict
hivm.hir.store (hivm::StoreOp)
HIVM data store operation

Syntax:

operation ::= `hivm.hir.store` `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              attr-dict
              (`atomic` `=` $atomic_kind^)?
              (`->` type($result_tensor)^)?
Stores the data on local buffer to global memory. Currently only support storing data on the unified buffer.

Examples:

hivm.store ins(%src : memref<16x16xf16, #hivm.address_space<ub>>) outs(%dst : memref<16x16xf16, #hivm.address_space<gm>>)
Constraints:

src and are expected to have the same element type.dst

If is set, the kind is one of , , .atomic_kindaddmaxmin

Traits: , , , AlwaysSpeculatableImplTraitOpPipeTrait<PIPE::PIPE_MTE3>SinglePipeOpTraitUniformReassociationFlattenTrait

Interfaces: , , , , , , , , , ConditionallySpeculatableCopyOpInterfaceDestinationStyleOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpInferCoreTypeInterfaceMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
atomic_kind	::mlir::hivm::AtomicKindAttr	
Atomic Operation Kind for StoreOp
may_implicit_transpose_with_last_axis	::mlir::BoolAttr	bool attribute
Operands:
Operand

Description

src

Tensor or Memref

dst

Tensor or Memref

Results:
Result

Description

result_tensor

ranked tensor of any type values

hivm.hir.sync_block (hivm::SyncBlockOp)
Hivm sync block between different kernels.

Syntax:

operation ::= `hivm.hir.sync_block` attr-dict `[` $sync_block_mode (`,` $flag_id^)?`]`
              (`ffts_base_addr` `=` $ffts_base_addr^)?
              (`tcube_pipe` `=` $tcube_pipe^)?
              (`tvector_pipe` `=` $tvector_pipe^)?
There are sync block modes:

ALL_CUBE : All cube are synchronized to a same point. needs to be set to the pipe that the cube core is waiting for.tcube_pipe

ALL_VECTOR : All vector are synchronized to a same point. needs to be set to the pipe that the vector core is waiting for.tvector_pipe

ALL_SUB_VECTOR : All sub-vector cores are synchronized to a same point.

BARRIER_CUBE : Used for cube-cube synchronization, it’s going to be lowered to a barrie.pipe_all and would only be copied the aic kernel.

BARRIER_VECTOR : Used for cube-cube synchronization, it’s going to be lowered to a barrie.pipe_all and would only be copied the aiv kernel.

ALL : All aic/aiv are synchronized to same point. needs to be set to the pipe that the vector core is waiting for.tvector_pipe

Note:

SyncBlockOp can only use after data is moved to gm.

$ffts_base_addr must be set in Ascend910B. Every time FFTS collect one specific from all subblocks, FFTS would set the flag ID back to the block in the group to do synchronization.$flag_id

Interfaces: InferCoreTypeInterface

Attributes:
Attribute	MLIR Type	Description
sync_block_mode	::mlir::hivm::SyncBlockModeAttr	
flag_id	::mlir::IntegerAttr	
An Attribute containing a integer value
tcube_pipe	::mlir::hivm::PipeAttr	
tvector_pipe	::mlir::hivm::PipeAttr	
Operands:
Operand

Description

ffts_base_addr

64-bit signless integer

hivm.hir.sync_block_lock (hivm::SyncBlockLockOp)
Sync block lock operation.

Syntax:

operation ::= `hivm.hir.sync_block_lock` attr-dict `lock_var` `(` $lock_var `:` type($lock_var) `)`
The sync_block_lock operation will not release until the lock_var equals the block idx. Example:

  hivm.hir.sync_block_lock lock_var(%lock : memref<1xi64>)
Operands:
Operand

Description

lock_var

1D memref of 64-bit signless integer values

hivm.hir.sync_block_set (hivm::SyncBlockSetOp)
Hivm set block sync.

Syntax:

operation ::= `hivm.hir.sync_block_set` attr-dict `[` $tcore_type `,` $tpipe `,` $pipe`]`
              `flag` `=` custom<FlagID>($static_flag_id, $dynamic_flag_id)
              (`ffts_base_addr` `=` $ffts_base_addr^)?
              (`sync_instr_mode` `=` $tsync_instr_mode^)?
Traits: AttrSizedOperandSegments

Interfaces: InferCoreTypeInterface

Attributes:
Attribute	MLIR Type	Description
tcore_type	::mlir::hivm::TCoreTypeAttr	
tpipe	::mlir::hivm::PipeAttr	
pipe	::mlir::hivm::PipeAttr	
static_flag_id	::mlir::IntegerAttr	
An Attribute containing a integer value
tsync_instr_mode	::mlir::hivm::SyncBlockInstrModeAttr	
Operands:
Operand

Description

dynamic_flag_id

64-bit signless integer

ffts_base_addr

64-bit signless integer

hivm.hir.sync_block_unlock (hivm::SyncBlockUnlockOp)
Sync block unlock operation.

Syntax:

operation ::= `hivm.hir.sync_block_unlock` attr-dict `lock_var` `(` $lock_var `:` type($lock_var) `)`
The operation will increase and release the lock_var. Example:sync_block_lock

  hivm.hir.sync_block_unlock lock_var(%lock : memref<1xi64>)
Operands:
Operand

Description

lock_var

1D memref of 64-bit signless integer values

hivm.hir.sync_block_wait (hivm::SyncBlockWaitOp)
Hivm wait block sync.

Syntax:

operation ::= `hivm.hir.sync_block_wait` attr-dict `[` $tcore_type `,` $tpipe `,` $pipe`]`
              `flag` `=` custom<FlagID>($static_flag_id, $dynamic_flag_id)
Interfaces: InferCoreTypeInterface

Attributes:
Attribute	MLIR Type	Description
tcore_type	::mlir::hivm::TCoreTypeAttr	
tpipe	::mlir::hivm::PipeAttr	
pipe	::mlir::hivm::PipeAttr	
static_flag_id	::mlir::IntegerAttr	
An Attribute containing a integer value
Operands:
Operand

Description

dynamic_flag_id

64-bit signless integer

hivm.hir.vabs (hivm::VAbsOp)
Elementwise Vector Absolute Value Op

Syntax:

operation ::= `hivm.hir.vabs` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
From the Elementwise Nary Vector Op template:

This operation performs element-wise operation on N operands and produces a single result. It may perform either transpose or broadcast along the way (but not both).

Common constraints:

Follows DestinationStyleOpInterface.

The number of input operands is N; the number of output/result is one.

The input/init operands and result have the same rank.

The first input is vector-only.

Additional constraints:

The input/init operands and result have the same element type.

Traits: , , , , , , , , , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsBroadcastableOTFCollapsibleConsecutiveTargetDimsTraitElementwiseNaryOpTrait<1>HIVMOpSameOperandsAndResultRankOpPipeTrait<PIPE::PIPE_V>SameOperandsElementTypeSinglePipeOpTraitTransposableOTFUniformReassociationFlattenTraitVectorCoreTypeTraitVectorOnlyTrait<0>

Interfaces: , , , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceExtraBufferOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpImplByScalarOpInterfaceMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
transpose	::mlir::DenseI64ArrayAttr	i64 dense array attribute
broadcast	::mlir::DenseI64ArrayAttr	i64 dense array attribute
Operands:
Operand

Description

src

variadic of shaped of any type values

dst

variadic of shaped of any type values

temp_buffer

memref of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vadd (hivm::VAddOp)
Elementwise Binary Vector Addition Op

Syntax:

operation ::= `hivm.hir.vadd` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
From the Elementwise Nary Vector Op template:

This operation performs element-wise operation on N operands and produces a single result. It may perform either transpose or broadcast along the way (but not both).

Common constraints:

Follows DestinationStyleOpInterface.

The number of input operands is N; the number of output/result is one.

The input/init operands and result have the same rank.

The first input is vector-only.

Additional constraints:

The input/init operands and result have the same element type.

Support both Vector-Vector and Vector-Scalar operation.

Traits: , , , , , , , , , , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsBroadcastableOTFCollapsibleConsecutiveTargetDimsTraitCommutativeOpTraitElementwiseNaryOpTrait<2>HIVMOpSameOperandsAndResultRankOpPipeTrait<PIPE::PIPE_V>SameOperandsElementTypeSinglePipeOpTraitTransposableOTFUniformReassociationFlattenTraitVectorCoreTypeTraitVectorOnlyTrait<0>

Interfaces: , , , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceExtraBufferOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpImplByScalarOpInterfaceMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
transpose	::mlir::DenseI64ArrayAttr	i64 dense array attribute
broadcast	::mlir::DenseI64ArrayAttr	i64 dense array attribute
Operands:
Operand

Description

src

variadic of any type

dst

variadic of shaped of any type values

temp_buffer

memref of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vand (hivm::VAndOp)
Elementwise Binary Vector And Op

Syntax:

operation ::= `hivm.hir.vand` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
From the Elementwise Nary Vector Op template:

This operation performs element-wise operation on N operands and produces a single result. It may perform either transpose or broadcast along the way (but not both).

Common constraints:

Follows DestinationStyleOpInterface.

The number of input operands is N; the number of output/result is one.

The input/init operands and result have the same rank.

The first input is vector-only.

Additional constraints:

The input/init operands and result have the same element type.

Support only Vector-Vector operation.

Traits: , , , , , , , , , , , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsBroadcastableOTFCollapsibleConsecutiveTargetDimsTraitCommutativeOpTraitElementwiseNaryOpTrait<2>HIVMOpSameOperandsAndResultRankOpPipeTrait<PIPE::PIPE_V>SameOperandsElementTypeSinglePipeOpTraitTransposableOTFUniformReassociationFlattenTraitVectorCoreTypeTraitVectorOnlyTrait<0>VectorOnlyTrait<1>

Interfaces: , , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceExtraBufferOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
transpose	::mlir::DenseI64ArrayAttr	i64 dense array attribute
broadcast	::mlir::DenseI64ArrayAttr	i64 dense array attribute
Operands:
Operand

Description

src

variadic of any type

dst

variadic of shaped of any type values

temp_buffer

memref of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.varange (hivm::VArangeOp)
Vector Arange Op

Syntax:

operation ::= `hivm.hir.varange` attr-dict
              (`offset` `[` $offset^ `]`)?
              `strides` `[` $strides `]`
              `outs` `(` $dst `:` type($dst) `)`
              (`->` type($result)^)?
Fill a vector with range 0,1,2… based on strides and offset. e.g. offset = 1, strides = [1, 2], tensor/memref shape = [2x4xi32], the result is [[1, 3, 5, 7, 2, 4, 6, 8]].

Constraints:

Must have at least one stride.

Default offset is 0.

Examples:

hivm.hir.varange offset[%o] strides[%s0, %s1] outs(%dst : memref<32xf32>)
%result = hivm.hir.varange offset[%o] strides[%s0, %s1] outs(%dst : tensor<32xf32>)
                            -> tensor<32xf32>
Traits: , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsOpPipeTrait<PIPE::PIPE_V>SinglePipeOpTraitVectorCoreTypeTrait

Interfaces: , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpMemoryEffectsOpInterfaceOpPipeInterface

Operands:
Operand

Description

dst

Tensor or Memref

offset

index

strides

variadic of index

Results:
Result

Description

result

ranked tensor of any type values

hivm.hir.vbrc (hivm::VBrcOp)
Vector Broadcast Op

Syntax:

operation ::= `hivm.hir.vbrc` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast_dims` `=` $broadcast_dims^)?
              (`->` type($result)^)?
Broadcast a vector or a scalar according to the broadcast axes array.

Constraints:

The input vector and output vector must have same rank and the same element type.

For the input operand, the size of the broadcasted axis must be 1.

The broadcast indices array cannot be empty for vector input.

The broadcast indices array must be empty for scalar input.

The broadcast indices array can not be larger than the ranks of the input vector.

The broadcast indices must be in .[0, RankOfSrcVec)

For i1 type, need to ensure that the tail axis of dst is aligned with 16, otherwise there will be a risk of memory stampede

Examples:

// Scalar broadcast
hivm.hir.vbrc ins(%src : i32) outs(%dst : memref<?xi32>)
// Vector broadcast
hivm.hir.vbrc ins(%src : memref<1xi32>) outs(%dst : memref<?xi32>) broadcast_dims = [0]
%result = hivm.hir.vbrc ins(%src : tensor<1xi32>) outs(%dst : tensor<?xi32>) broadcast_dims = [0] -> tensor<?xi32>
Traits: , , , , AlwaysSpeculatableImplTraitCollapsibleConsecutiveTargetDimsTraitSameOperandsElementTypeSinglePipeOpTraitUniformReassociationFlattenTrait

Interfaces: , , , , , , , , , , BiShengIRAggregatedOpInterfaceConditionallySpeculatableDestinationStyleOpInterfaceExtraBufferOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpInferCoreTypeInterfaceMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
broadcast_dims	::mlir::DenseI64ArrayAttr	i64 dense array attribute
Operands:
Operand

Description

src

any type

dst

Tensor or Memref

temp_buffer

memref of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vcast (hivm::VCastOp)
Elementwise Vector Type Conversion Op

Syntax:

operation ::= `hivm.hir.vcast` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`round_mode` `=` $round_mode^)?
              (`cast` `=` $cast^)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
From the Elementwise Nary Vector Op template:

This operation performs element-wise operation on N operands and produces a single result. It may perform either transpose or broadcast along the way (but not both).

Common constraints:

Follows DestinationStyleOpInterface.

The number of input operands is N; the number of output/result is one.

The input/init operands and result have the same rank.

The first input is vector-only.

Additional constraints:

Supports the following conversions:

| src  | dst  | roundingmode                                      |
|------|------|---------------------------------------------------|
| f32  | f32  | round, rint, floor, ceil, trunc                   |
| f32  | f16  | round, rint, floor, ceil, trunc, odd              |
| f32  | i64  | round, rint, floor, ceil, trunc                   |
| f32  | i32  | round, rint, floor, ceil, trunc                   |
| f32  | i16  | round, rint, floor, ceil, trunc                   |
| f32  | s64  | round, rint, floor, ceil, trunc                   |
| f32  | bf16 | round, rint, floor, ceil, trunc                   |
| f16  | f32  | rint                                              |
| f16  | i32  | round, rint, floor, ceil, trunc                   |
| f16  | i16  | round, rint, floor, ceil, trunc                   |
| f16  | i8   | round, rint, floor, ceil, trunc                   |
| f16  | ui8  | round, rint, floor, ceil, trunc                   |
| f16  | i4   | round, rint, floor, ceil, trunc                   |
| bf16 | f32  | rint                                              |
| bf16 | i32  | round, rint, floor, ceil, trunc                   |
| ui8  | f16  | rint                                              |
| i8   | f16  | rint                                              |
| i8   | i1   | rint                                              |
| i16  | f16  | round, rint, floor, ceil, trunc                   |
| i16  | f32  | rint                                              |
| i32  | f32  | round, rint, floor, ceil, trunc                   |
| i32  | i64  | rint                                              |
| i32  | i16  | rint                                              |
| i64  | i32  | rint                                              |
| i64  | f32  | round, rint, floor, ceil, trunc                   |
| i4   | f16  | rint                                              |
| i1   | f16  | rint                                              |
| i1   | f32  | rint                                              |
Traits: , , , , , , , , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsBroadcastableOTFCollapsibleConsecutiveTargetDimsTraitElementwiseNaryOpTrait<1>HIVMOpSameOperandsAndResultRankOpPipeTrait<PIPE::PIPE_V>SinglePipeOpTraitTransposableOTFUniformReassociationFlattenTraitVectorCoreTypeTraitVectorOnlyTrait<0>

Interfaces: , , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceExtraBufferOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
round_mode	::mlir::hivm::RoundModeAttr	
Round Mode for VCastOp
cast	::mlir::hivm::TypeFnAttr	
Cast for VCastOp
transpose	::mlir::DenseI64ArrayAttr	i64 dense array attribute
broadcast	::mlir::DenseI64ArrayAttr	i64 dense array attribute
Operands:
Operand

Description

src

variadic of any type

dst

variadic of shaped of any type values

temp_buffer

memref of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vcmp (hivm::VCmpOp)
Elementwise Binary Vector Comparison Op

Syntax:

operation ::= `hivm.hir.vcmp` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`compare_mode` `=` $compare_mode^)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
From the Elementwise Nary Vector Op template:

This operation performs element-wise operation on N operands and produces a single result. It may perform either transpose or broadcast along the way (but not both).

Common constraints:

Follows DestinationStyleOpInterface.

The number of input operands is N; the number of output/result is one.

The input/init operands and result have the same rank.

The first input is vector-only.

Compare elements from two source vector. If the comparison result is true, the corresponding bit of is 1 or 8.dst

Additional constraints:

The input vectors and output vector must have the same ranks

The element type of must be booldst

The input is vector-only.

Supports the following data type:

|    compare mode   |       element type      |
|-------------------|-------------------------|
| GE/GT/LE/LT/NE/EQ | f16, f32, i16, i32, i64 |
Traits: , , , , , , , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsCollapsibleConsecutiveTargetDimsTraitElementwiseNaryOpTrait<2>HIVMOpSameOperandsAndResultRankOpPipeTrait<PIPE::PIPE_V>SinglePipeOpTraitTransposableOTFUniformReassociationFlattenTraitVectorCoreTypeTraitVectorOnlyTrait<0>

Interfaces: , , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpImplByScalarOpInterfaceMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
compare_mode	::mlir::hivm::CompareModeAttr	
Compare Mode for VCmpOp
transpose	::mlir::DenseI64ArrayAttr	i64 dense array attribute
broadcast	::mlir::DenseI64ArrayAttr	i64 dense array attribute
Operands:
Operand

Description

src

variadic of any type

dst

variadic of shaped of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vconcat (hivm::VConcatOp)
Vector Concatenation Op

Syntax:

operation ::= `hivm.hir.vconcat` `dim` `(` $dim `)` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`->` type($result)^)?
The concat operation constructs a tensor out of a variadic list of input tensors, concatenated along a static dimension number. All inputs and the result type must share the same rank.

dim specifies the dimension along which to concatenate. The size of the concatenated dimension in the result must be equal to the sum of the sizes of the inputs along that dimension. All other dimensions in both the inputs and result must be the same size.

Example:

hivm.hir.vconcat dim(1) ins(%0, %1 : tensor<136x2048xf32>, tensor<136x2048xf32>)
                        outs(%2 : tensor<136x4096xf32>) -> tensor<136x4096xf32>
Traits: , , , , , AlwaysSpeculatableImplTraitOpPipeTrait<PIPE::PIPE_V>SameOperandsElementTypeSinglePipeOpTraitUniformReassociationFlattenTraitVectorCoreTypeTrait

Interfaces: , , , , , , , , BiShengIRAggregatedOpInterfaceConditionallySpeculatableDestinationStyleOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
dim	::mlir::IntegerAttr	64-bit signless integer attribute
Operands:
Operand

Description

src

variadic of any type

dst

Tensor or Memref

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vcos (hivm::VCosOp)
Elementwise Vector Cosine Op

Syntax:

operation ::= `hivm.hir.vcos` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
From the Elementwise Nary Vector Op template:

This operation performs element-wise operation on N operands and produces a single result. It may perform either transpose or broadcast along the way (but not both).

Common constraints:

Follows DestinationStyleOpInterface.

The number of input operands is N; the number of output/result is one.

The input/init operands and result have the same rank.

The first input is vector-only.

Additional constraints:

The input/init operands and result have the same element type.

Traits: , , , , , , , , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsCollapsibleConsecutiveTargetDimsTraitElementwiseNaryOpTrait<1>HIVMOpSameOperandsAndResultRankOpPipeTrait<PIPE::PIPE_V>SameOperandsElementTypeSinglePipeOpTraitTransposableOTFUniformReassociationFlattenTraitVectorCoreTypeTraitVectorOnlyTrait<0>

Interfaces: , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
transpose	::mlir::DenseI64ArrayAttr	i64 dense array attribute
broadcast	::mlir::DenseI64ArrayAttr	i64 dense array attribute
Operands:
Operand

Description

src

variadic of any type

dst

variadic of shaped of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vcumprod (hivm::VCumprodOp)
Vector Cumprod Op

Syntax:

operation ::= `hivm.hir.vcumprod` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `cum_dims` `=` $cum_dims
              `reverse` `=` $reverse
              (`->` type($result)^)?
Calculate the cumulative product of each element along the specified axis of . Each element along the specified axis in the output of cumprod contains the product of all elements from the first element to the current position in the original .srcsrc

Constraints:

The input vector and output vector must have the same rank and the same element type.

Arguments:

src: the tensor/memref from which to calculate the cumulative sum

dst: the tensor/memref to store elements

cum_dims: specifies the dimension along which to calculate the cumulative product.

reverse: specifies the direction of the cumulative product.

Examples:

hivm.hir.vcumprod ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>) cum_dims : [0] reverse = true
%result = hivm.hir.vcumprod ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) cum_dims : [0] reverse = true -> tensor<?xf32>
Traits: , , , , , AlwaysSpeculatableImplTraitOpPipeTrait<PIPE::PIPE_V>SameOperandsElementTypeSinglePipeOpTraitUniformReassociationFlattenTraitVectorCoreTypeTrait

Interfaces: , , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpImplByScalarOpInterfaceMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
cum_dims	::mlir::DenseI64ArrayAttr	i64 dense array attribute should be in increasing order
reverse	::mlir::BoolAttr	bool attribute
Operands:
Operand

Description

src

Tensor or Memref

dst

Tensor or Memref

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vcumsum (hivm::VCumsumOp)
Vector Cumsum Op

Syntax:

operation ::= `hivm.hir.vcumsum` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `cum_dims` `=` $cum_dims
              `reverse` `=` $reverse
              (`->` type($result)^)?
Calculate the cumulative sum of each element along the specified axis of . Each element along the specified axis in the output of cumsum contains the sum of all elements from the first element to the current position in the original .srcsrc

Constraints:

The input vector and output vector must have the same rank and the same element type.

Arguments:

src: the tensor/memref from which to calculate the cumulative sum

dst: the tensor/memref to store elements

cum_dims: specifies the dimension along which to calculate the cumulative sum.

reverse: specifies the direction of the cumulative sum.

Examples:

hivm.hir.vcumsum ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>) cum_dims : [0] reverse = true
%result = hivm.hir.vcumsum ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) cum_dims : [0] reverse = true -> tensor<?xf32>
Traits: , , , , , AlwaysSpeculatableImplTraitOpPipeTrait<PIPE::PIPE_V>SameOperandsElementTypeSinglePipeOpTraitUniformReassociationFlattenTraitVectorCoreTypeTrait

Interfaces: , , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpImplByScalarOpInterfaceMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
cum_dims	::mlir::DenseI64ArrayAttr	i64 dense array attribute should be in increasing order
reverse	::mlir::BoolAttr	bool attribute
Operands:
Operand

Description

src

Tensor or Memref

dst

Tensor or Memref

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vdeinterleave (hivm::VDeinterleaveOp)
Vector Deinterleave Op

Syntax:

operation ::= `hivm.hir.vdeinterleave` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`channel_num` `=` $channel_num^)?
              (`index_mode` `=` $index_mode^)?
              (`->` type($result)^)?
Deinterleave one tensor along the last dimension. The tensor’s last dimension size must be multiple of .channel_num

Traits: , , , , , , AlwaysSpeculatableImplTraitHIVMOpSameOperandsAndResultRankOpPipeTrait<PIPE::PIPE_V>SameOperandsElementTypeSinglePipeOpTraitUniformReassociationFlattenTraitVectorCoreTypeTrait

Interfaces: , , , , , , , , , BiShengIRAggregatedOpInterfaceConditionallySpeculatableDestinationStyleOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpImplByScalarOpInterfaceMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
channel_num	::mlir::IntegerAttr	64-bit signless integer attribute
index_mode	::mlir::hivm::DeinterleaveModeAttr	
HIVM deinterleave mode
Operands:
Operand

Description

src

Tensor or Memref

dst

variadic of Tensor or Memref

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vdiv (hivm::VDivOp)
Elementwise Binary Vector Division Op

Syntax:

operation ::= `hivm.hir.vdiv` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
From the Elementwise Nary Vector Op template:

This operation performs element-wise operation on N operands and produces a single result. It may perform either transpose or broadcast along the way (but not both).

Common constraints:

Follows DestinationStyleOpInterface.

The number of input operands is N; the number of output/result is one.

The input/init operands and result have the same rank.

The first input is vector-only.

Additional constraints:

The input/init operands and result have the same element type.

Support only Vector-Vector operation.

Traits: , , , , , , , , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsBroadcastableOTFCollapsibleConsecutiveTargetDimsTraitElementwiseNaryOpTrait<2>HIVMOpSameOperandsAndResultRankOpPipeTrait<PIPE::PIPE_V>SameOperandsElementTypeSinglePipeOpTraitTransposableOTFUniformReassociationFlattenTraitVectorCoreTypeTrait

Interfaces: , , , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceExtraBufferOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpImplByScalarOpInterfaceMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
transpose	::mlir::DenseI64ArrayAttr	i64 dense array attribute
broadcast	::mlir::DenseI64ArrayAttr	i64 dense array attribute
Operands:
Operand

Description

src

variadic of any type

dst

variadic of shaped of any type values

temp_buffer

memref of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.verf (hivm::VErfOp)
Elementwise Vector Error function Op

Syntax:

operation ::= `hivm.hir.verf` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
From the Elementwise Nary Vector Op template:

This operation performs element-wise operation on N operands and produces a single result. It may perform either transpose or broadcast along the way (but not both).

Common constraints:

Follows DestinationStyleOpInterface.

The number of input operands is N; the number of output/result is one.

The input/init operands and result have the same rank.

The first input is vector-only.

Additional constraints:

The input/init operands and result have the same element type.

Traits: , , , , , , , , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsCollapsibleConsecutiveTargetDimsTraitElementwiseNaryOpTrait<1>HIVMOpSameOperandsAndResultRankOpPipeTrait<PIPE::PIPE_V>SameOperandsElementTypeSinglePipeOpTraitTransposableOTFUniformReassociationFlattenTraitVectorCoreTypeTraitVectorOnlyTrait<0>

Interfaces: , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
transpose	::mlir::DenseI64ArrayAttr	i64 dense array attribute
broadcast	::mlir::DenseI64ArrayAttr	i64 dense array attribute
Operands:
Operand

Description

src

variadic of any type

dst

variadic of shaped of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vexp (hivm::VExpOp)
Elementwise Vector Exponential Op

Syntax:

operation ::= `hivm.hir.vexp` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
From the Elementwise Nary Vector Op template:

This operation performs element-wise operation on N operands and produces a single result. It may perform either transpose or broadcast along the way (but not both).

Common constraints:

Follows DestinationStyleOpInterface.

The number of input operands is N; the number of output/result is one.

The input/init operands and result have the same rank.

The first input is vector-only.

Additional constraints:

The input/init operands and result have the same element type.

Traits: , , , , , , , , , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsBroadcastableOTFCollapsibleConsecutiveTargetDimsTraitElementwiseNaryOpTrait<1>HIVMOpSameOperandsAndResultRankOpPipeTrait<PIPE::PIPE_V>SameOperandsElementTypeSinglePipeOpTraitTransposableOTFUniformReassociationFlattenTraitVectorCoreTypeTraitVectorOnlyTrait<0>

Interfaces: , , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceExtraBufferOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
transpose	::mlir::DenseI64ArrayAttr	i64 dense array attribute
broadcast	::mlir::DenseI64ArrayAttr	i64 dense array attribute
Operands:
Operand

Description

src

variadic of shaped of any type values

dst

variadic of shaped of any type values

temp_buffer

memref of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vflip (hivm::VFlipOp)
Vector Flip Op

Syntax:

operation ::= `hivm.hir.vflip` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `flip_axis` `=` $flip_axis
              (`->` type($result)^)?
Flips a tensor along the last dimension.

Traits: , , , , , AlwaysSpeculatableImplTraitOpPipeTrait<PIPE::PIPE_V>SameOperandsElementTypeSinglePipeOpTraitUniformReassociationFlattenTraitVectorCoreTypeTrait

Interfaces: , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
flip_axis	::mlir::IntegerAttr	64-bit signless integer attribute
Operands:
Operand

Description

src

Tensor or Memref

dst

Tensor or Memref

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vgather (hivm::VGatherOp)
Vector Gather Op

Syntax:

operation ::= `hivm.hir.vgather` attr-dict `ins` `(` $src `:` type($src) `)`
              `indices` `(` $indices `:` type($indices) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`->` type($result)^)?
Retrieve elements from a tensor/memref according to given indices, and store these elements in another tensor/memref. The gather axis is the last dimension.

Arguments:

src: the tensor/memref from which to gather elements

indices: the indices of elements to gather from src

dst: the tensor/memref to store elements

temp_buffer: extra memory required by gather op

Traits: , , , , , AlwaysSpeculatableImplTraitHIVMOpSameOperandsAndResultRankOpPipeTrait<PIPE::PIPE_V>SinglePipeOpTraitUniformReassociationFlattenTraitVectorCoreTypeTrait

Interfaces: , , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceExtraBufferOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpMemoryEffectsOpInterfaceOpPipeInterface

Operands:
Operand

Description

src

Tensor or Memref

indices

Tensor or Memref

dst

Tensor or Memref

temp_buffer

memref of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vinterleave (hivm::VInterleaveOp)
Vetor Interleave Op

Syntax:

operation ::= `hivm.hir.vinterleave` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `interleave_channel_nums` `=` $interleave_channel_nums
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`->` type($result)^)?
Interleaves the values of tensors along their last dimension. All tensors must have the same shape.N

Traits: , , , , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsHIVMOpSameOperandsAndResultRankOpPipeTrait<PIPE::PIPE_V>SameOperandsElementTypeSinglePipeOpTraitUniformReassociationFlattenTraitVectorCoreTypeTrait

Interfaces: , , , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceExtraBufferOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpImplByScalarOpInterfaceMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
interleave_channel_nums	::mlir::IntegerAttr	64-bit signless integer attribute
Operands:
Operand

Description

src

variadic of any type

dst

Tensor or Memref

temp_buffer

memref of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vln (hivm::VLnOp)
Elementwise Vector Natural Logarithm Op

Syntax:

operation ::= `hivm.hir.vln` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
From the Elementwise Nary Vector Op template:

This operation performs element-wise operation on N operands and produces a single result. It may perform either transpose or broadcast along the way (but not both).

Common constraints:

Follows DestinationStyleOpInterface.

The number of input operands is N; the number of output/result is one.

The input/init operands and result have the same rank.

The first input is vector-only.

Additional constraints:

The input/init operands and result have the same element type.

Traits: , , , , , , , , , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsBroadcastableOTFCollapsibleConsecutiveTargetDimsTraitElementwiseNaryOpTrait<1>HIVMOpSameOperandsAndResultRankOpPipeTrait<PIPE::PIPE_V>SameOperandsElementTypeSinglePipeOpTraitTransposableOTFUniformReassociationFlattenTraitVectorCoreTypeTraitVectorOnlyTrait<0>

Interfaces: , , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceExtraBufferOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
transpose	::mlir::DenseI64ArrayAttr	i64 dense array attribute
broadcast	::mlir::DenseI64ArrayAttr	i64 dense array attribute
Operands:
Operand

Description

src

variadic of shaped of any type values

dst

variadic of shaped of any type values

temp_buffer

memref of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vmax (hivm::VMaxOp)
Elementwise Binary Vector Maximum Op

Syntax:

operation ::= `hivm.hir.vmax` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
From the Elementwise Nary Vector Op template:

This operation performs element-wise operation on N operands and produces a single result. It may perform either transpose or broadcast along the way (but not both).

Common constraints:

Follows DestinationStyleOpInterface.

The number of input operands is N; the number of output/result is one.

The input/init operands and result have the same rank.

The first input is vector-only.

Additional constraints:

The input/init operands and result have the same element type.

Support both Vector-Vector and Vector-Scalar operation.

Traits: , , , , , , , , , , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsBroadcastableOTFCollapsibleConsecutiveTargetDimsTraitCommutativeOpTraitElementwiseNaryOpTrait<2>HIVMOpSameOperandsAndResultRankOpPipeTrait<PIPE::PIPE_V>SameOperandsElementTypeSinglePipeOpTraitTransposableOTFUniformReassociationFlattenTraitVectorCoreTypeTraitVectorOnlyTrait<0>

Interfaces: , , , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceExtraBufferOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpImplByScalarOpInterfaceMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
transpose	::mlir::DenseI64ArrayAttr	i64 dense array attribute
broadcast	::mlir::DenseI64ArrayAttr	i64 dense array attribute
Operands:
Operand

Description

src

variadic of any type

dst

variadic of shaped of any type values

temp_buffer

memref of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vmin (hivm::VMinOp)
Elementwise Binary Vector Minimum Op

Syntax:

operation ::= `hivm.hir.vmin` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
From the Elementwise Nary Vector Op template:

This operation performs element-wise operation on N operands and produces a single result. It may perform either transpose or broadcast along the way (but not both).

Common constraints:

Follows DestinationStyleOpInterface.

The number of input operands is N; the number of output/result is one.

The input/init operands and result have the same rank.

The first input is vector-only.

Additional constraints:

The input/init operands and result have the same element type.

Support both Vector-Vector and Vector-Scalar operation.

Traits: , , , , , , , , , , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsBroadcastableOTFCollapsibleConsecutiveTargetDimsTraitCommutativeOpTraitElementwiseNaryOpTrait<2>HIVMOpSameOperandsAndResultRankOpPipeTrait<PIPE::PIPE_V>SameOperandsElementTypeSinglePipeOpTraitTransposableOTFUniformReassociationFlattenTraitVectorCoreTypeTraitVectorOnlyTrait<0>

Interfaces: , , , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceExtraBufferOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpImplByScalarOpInterfaceMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
transpose	::mlir::DenseI64ArrayAttr	i64 dense array attribute
broadcast	::mlir::DenseI64ArrayAttr	i64 dense array attribute
Operands:
Operand

Description

src

variadic of any type

dst

variadic of shaped of any type values

temp_buffer

memref of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vmod (hivm::VModOp)
Elementwise Vector Mod Op

Syntax:

operation ::= `hivm.hir.vmod` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
From the Elementwise Nary Vector Op template:

This operation performs element-wise operation on N operands and produces a single result. It may perform either transpose or broadcast along the way (but not both).

Common constraints:

Follows DestinationStyleOpInterface.

The number of input operands is N; the number of output/result is one.

The input/init operands and result have the same rank.

The first input is vector-only.

Additional constraints:

The input/init operands and result have the same element type.

Traits: , , , , , , , , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsCollapsibleConsecutiveTargetDimsTraitElementwiseNaryOpTrait<2>HIVMOpSameOperandsAndResultRankOpPipeTrait<PIPE::PIPE_V>SameOperandsElementTypeSinglePipeOpTraitTransposableOTFUniformReassociationFlattenTraitVectorCoreTypeTraitVectorOnlyTrait<0>

Interfaces: , , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpImplByScalarOpInterfaceMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
transpose	::mlir::DenseI64ArrayAttr	i64 dense array attribute
broadcast	::mlir::DenseI64ArrayAttr	i64 dense array attribute
Operands:
Operand

Description

src

variadic of any type

dst

variadic of shaped of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vmul (hivm::VMulOp)
Elementwise Binary Vector Multiplication Op

Syntax:

operation ::= `hivm.hir.vmul` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
From the Elementwise Nary Vector Op template:

This operation performs element-wise operation on N operands and produces a single result. It may perform either transpose or broadcast along the way (but not both).

Common constraints:

Follows DestinationStyleOpInterface.

The number of input operands is N; the number of output/result is one.

The input/init operands and result have the same rank.

The first input is vector-only.

Additional constraints:

The input/init operands and result have the same element type.

Support both Vector-Vector and Vector-Scalar operation.

Traits: , , , , , , , , , , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsBroadcastableOTFCollapsibleConsecutiveTargetDimsTraitCommutativeOpTraitElementwiseNaryOpTrait<2>HIVMOpSameOperandsAndResultRankOpPipeTrait<PIPE::PIPE_V>SameOperandsElementTypeSinglePipeOpTraitTransposableOTFUniformReassociationFlattenTraitVectorCoreTypeTraitVectorOnlyTrait<0>

Interfaces: , , , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceExtraBufferOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpImplByScalarOpInterfaceMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
transpose	::mlir::DenseI64ArrayAttr	i64 dense array attribute
broadcast	::mlir::DenseI64ArrayAttr	i64 dense array attribute
Operands:
Operand

Description

src

variadic of any type

dst

variadic of shaped of any type values

temp_buffer

memref of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vmulext (hivm::VMulExtOp)
Elementwise Binary Vector Multiplication that Calculates the Most Significant 32-bits.

Syntax:

operation ::= `hivm.hir.vmulext` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
From the Elementwise Nary Vector Op template:

This operation performs element-wise operation on N operands and produces a single result. It may perform either transpose or broadcast along the way (but not both).

Common constraints:

Follows DestinationStyleOpInterface.

The number of input operands is N; the number of output/result is one.

The input/init operands and result have the same rank.

The first input is vector-only.

Additional constraints:

The input/init operands and result have the same element type.

Support Vector-Vector operation.

Traits: , , , , , , , , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsCollapsibleConsecutiveTargetDimsTraitElementwiseNaryOpTrait<2>HIVMOpSameOperandsAndResultRankOpPipeTrait<PIPE::PIPE_V>SameOperandsElementTypeSinglePipeOpTraitTransposableOTFUniformReassociationFlattenTraitVectorCoreTypeTraitVectorOnlyTrait<0>

Interfaces: , , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpImplByScalarOpInterfaceMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
transpose	::mlir::DenseI64ArrayAttr	i64 dense array attribute
broadcast	::mlir::DenseI64ArrayAttr	i64 dense array attribute
Operands:
Operand

Description

src

variadic of any type

dst

variadic of shaped of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vmulextended (hivm::VMulextendedOp)
Vector Mulextended Op

Syntax:

operation ::= `hivm.hir.vmulextended` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`->` type($result)^)?
Do vmul on two tensors. Get both high and low 16-bits.

Traits: , , , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsHIVMOpSameOperandsAndResultRankOpPipeTrait<PIPE::PIPE_V>SinglePipeOpTraitUniformReassociationFlattenTraitVectorCoreTypeTrait

Interfaces: , , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceExtraBufferOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpMemoryEffectsOpInterfaceOpPipeInterface

Operands:
Operand

Description

src

variadic of Tensor or Memref

dst

variadic of Tensor or Memref

temp_buffer

memref of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vnot (hivm::VNotOp)
Elementwise Vector Not Op

Syntax:

operation ::= `hivm.hir.vnot` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
From the Elementwise Nary Vector Op template:

This operation performs element-wise operation on N operands and produces a single result. It may perform either transpose or broadcast along the way (but not both).

Common constraints:

Follows DestinationStyleOpInterface.

The number of input operands is N; the number of output/result is one.

The input/init operands and result have the same rank.

The first input is vector-only.

Additional constraints:

The input/init operands and result have the same element type.

Traits: , , , , , , , , , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsBroadcastableOTFCollapsibleConsecutiveTargetDimsTraitElementwiseNaryOpTrait<1>HIVMOpSameOperandsAndResultRankOpPipeTrait<PIPE::PIPE_V>SameOperandsElementTypeSinglePipeOpTraitTransposableOTFUniformReassociationFlattenTraitVectorCoreTypeTraitVectorOnlyTrait<0>

Interfaces: , , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceExtraBufferOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
transpose	::mlir::DenseI64ArrayAttr	i64 dense array attribute
broadcast	::mlir::DenseI64ArrayAttr	i64 dense array attribute
Operands:
Operand

Description

src

variadic of shaped of any type values

dst

variadic of shaped of any type values

temp_buffer

memref of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vor (hivm::VOrOp)
Elementwise Binary Vector Or Op

Syntax:

operation ::= `hivm.hir.vor` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
From the Elementwise Nary Vector Op template:

This operation performs element-wise operation on N operands and produces a single result. It may perform either transpose or broadcast along the way (but not both).

Common constraints:

Follows DestinationStyleOpInterface.

The number of input operands is N; the number of output/result is one.

The input/init operands and result have the same rank.

The first input is vector-only.

Additional constraints:

The input/init operands and result have the same element type.

Support only Vector-Vector operation.

Traits: , , , , , , , , , , , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsBroadcastableOTFCollapsibleConsecutiveTargetDimsTraitCommutativeOpTraitElementwiseNaryOpTrait<2>HIVMOpSameOperandsAndResultRankOpPipeTrait<PIPE::PIPE_V>SameOperandsElementTypeSinglePipeOpTraitTransposableOTFUniformReassociationFlattenTraitVectorCoreTypeTraitVectorOnlyTrait<0>VectorOnlyTrait<1>

Interfaces: , , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceExtraBufferOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
transpose	::mlir::DenseI64ArrayAttr	i64 dense array attribute
broadcast	::mlir::DenseI64ArrayAttr	i64 dense array attribute
Operands:
Operand

Description

src

variadic of any type

dst

variadic of shaped of any type values

temp_buffer

memref of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vpad (hivm::VPadOp)
Vector Pad Op

Syntax:

operation ::= `hivm.hir.vpad` attr-dict
              `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `low` `` custom<DynamicIndexList>($low, $static_low)
              `high` `` custom<DynamicIndexList>($high, $static_high)
              `pad_value` $pad_value `:` type($pad_value)
              (`->` type($result)^)?
Pads the input operand. Operation semantic is similar to .tensor.pad

Arguments:

src: the tensor/memref on which to pad values

dst: reserved for bufferization

pad_value: the value to pad

low: the padding lengths along the start of each dimension

high: the padding lengths along the end of each dimension

Example:

hivm.hir.vpad ins(%src : tensor<2x16xf32>) outs(%dst: tensor<?x16xf32>)
              low[%first_dim_low, 0] high[%first_dim_high, 0]
              pad_value %pad_value : f32
                -> tensor<?x16xf32>
Traits: , , , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsHIVMOpSameOperandsAndResultRankOpPipeTrait<PIPE::PIPE_V>SinglePipeOpTraitUniformReassociationFlattenTraitVectorCoreTypeTrait

Interfaces: , , , , , , , , BiShengIRAggregatedOpInterfaceConditionallySpeculatableDestinationStyleOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
static_low	::mlir::DenseI64ArrayAttr	i64 dense array attribute
static_high	::mlir::DenseI64ArrayAttr	i64 dense array attribute
Operands:
Operand

Description

src

Tensor or Memref

dst

Tensor or Memref

pad_value

any type

low

variadic of index

high

variadic of index

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vpow (hivm::VPowOp)
Elementwise Binary Vector Power Op

Syntax:

operation ::= `hivm.hir.vpow` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
From the Elementwise Nary Vector Op template:

This operation performs element-wise operation on N operands and produces a single result. It may perform either transpose or broadcast along the way (but not both).

Common constraints:

Follows DestinationStyleOpInterface.

The number of input operands is N; the number of output/result is one.

The input/init operands and result have the same rank.

The first input is vector-only.

Additional constraints:

The input/init operands and result have the same element type.

Support both Vector-Vector and Vector-Scalar operation.

Traits: , , , , , , , , , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsCollapsibleConsecutiveTargetDimsTraitElementwiseNaryOpTrait<2>HIVMOpSameOperandsAndResultRankOpPipeTrait<PIPE::PIPE_V>SameOperandsElementTypeSinglePipeOpTraitTransposableOTFUniformReassociationFlattenTraitVectorCoreTypeTraitVectorOnlyTrait<0>VectorOnlyTrait<1>

Interfaces: , , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceExtraBufferOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
transpose	::mlir::DenseI64ArrayAttr	i64 dense array attribute
broadcast	::mlir::DenseI64ArrayAttr	i64 dense array attribute
Operands:
Operand

Description

src

variadic of any type

dst

variadic of shaped of any type values

temp_buffer

memref of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vrec (hivm::VRecOp)
Elementwise Vector Reciprocal Op

Syntax:

operation ::= `hivm.hir.vrec` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
From the Elementwise Nary Vector Op template:

This operation performs element-wise operation on N operands and produces a single result. It may perform either transpose or broadcast along the way (but not both).

Common constraints:

Follows DestinationStyleOpInterface.

The number of input operands is N; the number of output/result is one.

The input/init operands and result have the same rank.

The first input is vector-only.

Additional constraints:

The input/init operands and result have the same element type.

Traits: , , , , , , , , , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsBroadcastableOTFCollapsibleConsecutiveTargetDimsTraitElementwiseNaryOpTrait<1>HIVMOpSameOperandsAndResultRankOpPipeTrait<PIPE::PIPE_V>SameOperandsElementTypeSinglePipeOpTraitTransposableOTFUniformReassociationFlattenTraitVectorCoreTypeTraitVectorOnlyTrait<0>

Interfaces: , , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceExtraBufferOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
transpose	::mlir::DenseI64ArrayAttr	i64 dense array attribute
broadcast	::mlir::DenseI64ArrayAttr	i64 dense array attribute
Operands:
Operand

Description

src

variadic of shaped of any type values

dst

variadic of shaped of any type values

temp_buffer

memref of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vreduce (hivm::VReduceOp)
Vector Reduction Op

Syntax:

operation ::= `hivm.hir.vreduce` attr-dict $arith `ins` `(` $src `:` type($src) `)`
              (`indices` `(` $indices^ `:` type($indices) `)`)?
              `outs` `(` $dst `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              `reduce_dims` `=` $reduce_dims
              (`->` type($result)^)?
Recuce one or more axes of the source vector according to the reduction axes array, starting from an init value.

Constraints:

The input vector and output vector must have the same rank and the same element type.

For the output operand, the size of the reduced axis must be 1.

The reduction indices array can not be empty, nor can be larger than the ranks of the input vector.

The reduced indices must be in .[0, RankOfDstVec)

Examples:

hivm.hir.vreduce <add> ins(%src : memref<?xf32>) outs(%dst : memref<1xf32>) reduce_dims : [1]
%result = hivm.hir.vreduce <max> ins(%src : tensor<?xf32>) outs(%dst : tensor<1xf32>) reduce_dims : [0] -> tensor<1xf32>
Traits: , , , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsCollapsibleConsecutiveTargetDimsTraitOpPipeTrait<PIPE::PIPE_V>SinglePipeOpTraitUniformReassociationFlattenTraitVectorCoreTypeTrait

Interfaces: , , , , , , , , , , BiShengIRAggregatedOpInterfaceConditionallySpeculatableDestinationStyleOpInterfaceExtraBufferOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpImplByScalarOpInterfaceMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
arith	::mlir::hivm::ReduceOpAttr	
reduce_dims	::mlir::DenseI64ArrayAttr	i64 dense array attribute
Operands:
Operand

Description

src

Tensor or Memref

dst

variadic of Tensor or Memref

temp_buffer

memref of any type values

indices

Tensor or Memref

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vrelu (hivm::VReluOp)
Elementwise Vector Rectified Linear Unit Op

Syntax:

operation ::= `hivm.hir.vrelu` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
From the Elementwise Nary Vector Op template:

This operation performs element-wise operation on N operands and produces a single result. It may perform either transpose or broadcast along the way (but not both).

Common constraints:

Follows DestinationStyleOpInterface.

The number of input operands is N; the number of output/result is one.

The input/init operands and result have the same rank.

The first input is vector-only.

Additional constraints:

The input/init operands and result have the same element type.

Traits: , , , , , , , , , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsBroadcastableOTFCollapsibleConsecutiveTargetDimsTraitElementwiseNaryOpTrait<1>HIVMOpSameOperandsAndResultRankOpPipeTrait<PIPE::PIPE_V>SameOperandsElementTypeSinglePipeOpTraitTransposableOTFUniformReassociationFlattenTraitVectorCoreTypeTraitVectorOnlyTrait<0>

Interfaces: , , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceExtraBufferOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
transpose	::mlir::DenseI64ArrayAttr	i64 dense array attribute
broadcast	::mlir::DenseI64ArrayAttr	i64 dense array attribute
Operands:
Operand

Description

src

variadic of shaped of any type values

dst

variadic of shaped of any type values

temp_buffer

memref of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vrsqrt (hivm::VRsqrtOp)
Elementwise Vector Reciprocal Square Root Op

Syntax:

operation ::= `hivm.hir.vrsqrt` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
From the Elementwise Nary Vector Op template:

This operation performs element-wise operation on N operands and produces a single result. It may perform either transpose or broadcast along the way (but not both).

Common constraints:

Follows DestinationStyleOpInterface.

The number of input operands is N; the number of output/result is one.

The input/init operands and result have the same rank.

The first input is vector-only.

Additional constraints:

The input/init operands and result have the same element type.

Traits: , , , , , , , , , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsBroadcastableOTFCollapsibleConsecutiveTargetDimsTraitElementwiseNaryOpTrait<1>HIVMOpSameOperandsAndResultRankOpPipeTrait<PIPE::PIPE_V>SameOperandsElementTypeSinglePipeOpTraitTransposableOTFUniformReassociationFlattenTraitVectorCoreTypeTraitVectorOnlyTrait<0>

Interfaces: , , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceExtraBufferOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
transpose	::mlir::DenseI64ArrayAttr	i64 dense array attribute
broadcast	::mlir::DenseI64ArrayAttr	i64 dense array attribute
Operands:
Operand

Description

src

variadic of shaped of any type values

dst

variadic of shaped of any type values

temp_buffer

memref of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vsel (hivm::VSelOp)
Elementwise Vector Selection Op

Syntax:

operation ::= `hivm.hir.vsel` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
From the Elementwise Nary Vector Op template:

This operation performs element-wise operation on N operands and produces a single result. It may perform either transpose or broadcast along the way (but not both).

Common constraints:

Follows DestinationStyleOpInterface.

The number of input operands is N; the number of output/result is one.

The input/init operands and result have the same rank.

The first input is vector-only.

Select elements from two source vector according to the binary vector. If the corresponding bit of the indicator is 1, select . Otherwise, select .conditionsrc0src1

Additional constraints:

The input vectors and output vector must have the same ranks.

The element type of indicator vector must be bool.

Traits: , , , , , , , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsBroadcastableOTFCollapsibleConsecutiveTargetDimsTraitElementwiseNaryOpTrait<3>HIVMOpSameOperandsAndResultRankOpPipeTrait<PIPE::PIPE_V>SinglePipeOpTraitTransposableOTFUniformReassociationFlattenTraitVectorCoreTypeTrait

Interfaces: , , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceExtraBufferOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
transpose	::mlir::DenseI64ArrayAttr	i64 dense array attribute
broadcast	::mlir::DenseI64ArrayAttr	i64 dense array attribute
Operands:
Operand

Description

src

variadic of any type

dst

variadic of shaped of any type values

temp_buffer

memref of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vshl (hivm::VShLOp)
Elementwise Binary Vector Shift Left Op

Syntax:

operation ::= `hivm.hir.vshl` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
From the Elementwise Nary Vector Op template:

This operation performs element-wise operation on N operands and produces a single result. It may perform either transpose or broadcast along the way (but not both).

Common constraints:

Follows DestinationStyleOpInterface.

The number of input operands is N; the number of output/result is one.

The input/init operands and result have the same rank.

The first input is vector-only.

Additional constraints:

The input vector and result have the same element type.

Support only Vector - Scalar operation.

Traits: , , , , , , , , , , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsBroadcastableOTFCollapsibleConsecutiveTargetDimsTraitElementwiseNaryOpTrait<2>HIVMOpSameOperandsAndResultRankOpPipeTrait<PIPE::PIPE_V>SameOperandsElementTypeScalarOnlyHWTrait<1>SinglePipeOpTraitTransposableOTFUniformReassociationFlattenTraitVectorCoreTypeTraitVectorOnlyTrait<0>

Interfaces: , , , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceExtraBufferOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpImplByScalarOpInterfaceMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
transpose	::mlir::DenseI64ArrayAttr	i64 dense array attribute
broadcast	::mlir::DenseI64ArrayAttr	i64 dense array attribute
Operands:
Operand

Description

src

variadic of any type

dst

variadic of shaped of any type values

temp_buffer

memref of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vshr (hivm::VShROp)
Elementwise Binary Vector Shift Right Op

Syntax:

operation ::= `hivm.hir.vshr` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`round` `:` $round^ )?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
From the Elementwise Nary Vector Op template:

This operation performs element-wise operation on N operands and produces a single result. It may perform either transpose or broadcast along the way (but not both).

Common constraints:

Follows DestinationStyleOpInterface.

The number of input operands is N; the number of output/result is one.

The input/init operands and result have the same rank.

The first input is vector-only.

Additional constraints:

The input vector and result have the same element type.

Support only Vector - Scalar operation.

If is set to true, rounding is applied during arithmetic shift right.round

Traits: , , , , , , , , , , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsBroadcastableOTFCollapsibleConsecutiveTargetDimsTraitElementwiseNaryOpTrait<2>HIVMOpSameOperandsAndResultRankOpPipeTrait<PIPE::PIPE_V>SameOperandsElementTypeScalarOnlyHWTrait<1>SinglePipeOpTraitTransposableOTFUniformReassociationFlattenTraitVectorCoreTypeTraitVectorOnlyTrait<0>

Interfaces: , , , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceExtraBufferOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpImplByScalarOpInterfaceMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
round	::mlir::BoolAttr	bool attribute
transpose	::mlir::DenseI64ArrayAttr	i64 dense array attribute
broadcast	::mlir::DenseI64ArrayAttr	i64 dense array attribute
Operands:
Operand

Description

src

variadic of any type

dst

variadic of shaped of any type values

temp_buffer

memref of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vsin (hivm::VSinOp)
Elementwise Vector Sine Op

Syntax:

operation ::= `hivm.hir.vsin` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
From the Elementwise Nary Vector Op template:

This operation performs element-wise operation on N operands and produces a single result. It may perform either transpose or broadcast along the way (but not both).

Common constraints:

Follows DestinationStyleOpInterface.

The number of input operands is N; the number of output/result is one.

The input/init operands and result have the same rank.

The first input is vector-only.

Additional constraints:

The input/init operands and result have the same element type.

Traits: , , , , , , , , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsCollapsibleConsecutiveTargetDimsTraitElementwiseNaryOpTrait<1>HIVMOpSameOperandsAndResultRankOpPipeTrait<PIPE::PIPE_V>SameOperandsElementTypeSinglePipeOpTraitTransposableOTFUniformReassociationFlattenTraitVectorCoreTypeTraitVectorOnlyTrait<0>

Interfaces: , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
transpose	::mlir::DenseI64ArrayAttr	i64 dense array attribute
broadcast	::mlir::DenseI64ArrayAttr	i64 dense array attribute
Operands:
Operand

Description

src

variadic of any type

dst

variadic of shaped of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vsort (hivm::VSortOp)
Vector Sort Op

Syntax:

operation ::= `hivm.hir.vsort` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              `descending` `=` $descending
              `sort_axis` `=` $sort_axis
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`->` type($result)^)?
Sort the sorting axis of in ascending or descending order, and output the sorted value and the index corresponding to the value.src

Constraints:

The input vector and output vector must have the same rank.

Currently only tail axis sorting is supported.

Arguments:

src: the tensor/memref from which to be sorted

dst_value: the tensor/memref to store the sorted value

dst_index: the tensor/memref to store the index corresponding to dst_value

descending: determines whether to sort in ascending or descending order. The default is false, which means ascending order

sort_axis: Axis to be sorted

Examples:

hivm.hir.vsort ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>) descending = true sort_axis = 0
%result = hivm.hir.vsort ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) descending = true sort_axis = 0 -> tensor<?xf32>
Traits: , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsOpPipeTrait<PIPE::PIPE_V>SinglePipeOpTraitVectorCoreTypeTrait

Interfaces: , , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceExtraBufferOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
descending	::mlir::BoolAttr	bool attribute
sort_axis	::mlir::IntegerAttr	64-bit signless integer attribute
Operands:
Operand

Description

src

Tensor or Memref

dst

variadic of Tensor or Memref

temp_buffer

memref of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vsqrt (hivm::VSqrtOp)
Elementwise Vector Square Root Op

Syntax:

operation ::= `hivm.hir.vsqrt` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst  `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
From the Elementwise Nary Vector Op template:

This operation performs element-wise operation on N operands and produces a single result. It may perform either transpose or broadcast along the way (but not both).

Common constraints:

Follows DestinationStyleOpInterface.

The number of input operands is N; the number of output/result is one.

The input/init operands and result have the same rank.

The first input is vector-only.

Additional constraints:

The input/init operands and result have the same element type.

Traits: , , , , , , , , , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsBroadcastableOTFCollapsibleConsecutiveTargetDimsTraitElementwiseNaryOpTrait<1>HIVMOpSameOperandsAndResultRankOpPipeTrait<PIPE::PIPE_V>SameOperandsElementTypeSinglePipeOpTraitTransposableOTFUniformReassociationFlattenTraitVectorCoreTypeTraitVectorOnlyTrait<0>

Interfaces: , , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceExtraBufferOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
transpose	::mlir::DenseI64ArrayAttr	i64 dense array attribute
broadcast	::mlir::DenseI64ArrayAttr	i64 dense array attribute
Operands:
Operand

Description

src

variadic of shaped of any type values

dst

variadic of shaped of any type values

temp_buffer

memref of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vsub (hivm::VSubOp)
Elementwise Binary Vector Subtraction Op

Syntax:

operation ::= `hivm.hir.vsub` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
From the Elementwise Nary Vector Op template:

This operation performs element-wise operation on N operands and produces a single result. It may perform either transpose or broadcast along the way (but not both).

Common constraints:

Follows DestinationStyleOpInterface.

The number of input operands is N; the number of output/result is one.

The input/init operands and result have the same rank.

The first input is vector-only.

Additional constraints:

The input/init operands and result have the same element type.

Support both Vector-Vector and Vector-Scalar operation.

Traits: , , , , , , , , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsBroadcastableOTFCollapsibleConsecutiveTargetDimsTraitElementwiseNaryOpTrait<2>HIVMOpSameOperandsAndResultRankOpPipeTrait<PIPE::PIPE_V>SameOperandsElementTypeSinglePipeOpTraitTransposableOTFUniformReassociationFlattenTraitVectorCoreTypeTrait

Interfaces: , , , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceExtraBufferOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpImplByScalarOpInterfaceMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
transpose	::mlir::DenseI64ArrayAttr	i64 dense array attribute
broadcast	::mlir::DenseI64ArrayAttr	i64 dense array attribute
Operands:
Operand

Description

src

variadic of any type

dst

variadic of shaped of any type values

temp_buffer

memref of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vtanh (hivm::VTanhOp)
Elementwise Vector Hyperbolic Tangent Op

Syntax:

operation ::= `hivm.hir.vtanh` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
From the Elementwise Nary Vector Op template:

This operation performs element-wise operation on N operands and produces a single result. It may perform either transpose or broadcast along the way (but not both).

Common constraints:

Follows DestinationStyleOpInterface.

The number of input operands is N; the number of output/result is one.

The input/init operands and result have the same rank.

The first input is vector-only.

Additional constraints:

The input/init operands and result have the same element type.

Traits: , , , , , , , , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsCollapsibleConsecutiveTargetDimsTraitElementwiseNaryOpTrait<1>HIVMOpSameOperandsAndResultRankOpPipeTrait<PIPE::PIPE_V>SameOperandsElementTypeSinglePipeOpTraitTransposableOTFUniformReassociationFlattenTraitVectorCoreTypeTraitVectorOnlyTrait<0>

Interfaces: , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
transpose	::mlir::DenseI64ArrayAttr	i64 dense array attribute
broadcast	::mlir::DenseI64ArrayAttr	i64 dense array attribute
Operands:
Operand

Description

src

variadic of any type

dst

variadic of shaped of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vtranspose (hivm::VTransposeOp)
Vector Transpose Op

Syntax:

operation ::= `hivm.hir.vtranspose` attr-dict `ins` `(` $src `:` type($src) `)`
              `outs` `(` $dst `:` type($dst) `)`
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`permutation` `=` $permutation^)?
              (`disable_align` `=` $disable_align^)?
              (`->` type($result)^)?
Permutes the dimensions of ‘src’ according to the given . In other words: .permutationdim(dst, i) = dim(src, permutation[i])

Constraints:

The input vector and output vector must have same rank, and the same element type.

Examples:

 hivm.hir.vtranspose ins(%src : memref<32x8xf32>) outs(%dst : memref<8x32xf32>) permutation = [1, 0]
 %result = hivm.hir.vtranspose ins(%src : tensor<32x8xf32>) outs(%dst: tensor<8x32xf32>) permutation = [1, 0] -> tensor<8x32xf32>
Traits: , , , , AlwaysSpeculatableImplTraitOpPipeTrait<PIPE::PIPE_V>SinglePipeOpTraitUniformReassociationFlattenTraitVectorCoreTypeTrait

Interfaces: , , , , , , , , , BiShengIRAggregatedOpInterfaceConditionallySpeculatableDestinationStyleOpInterfaceExtraBufferOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
permutation	::mlir::DenseI64ArrayAttr	i64 dense array attribute
disable_align	::mlir::BoolAttr	bool attribute
Operands:
Operand

Description

src

Tensor or Memref

dst

Tensor or Memref

temp_buffer

memref of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.vxor (hivm::VXorOp)
Elementwise Binary Vector Xor Op

Syntax:

operation ::= `hivm.hir.vxor` attr-dict (`ins` `(` $src^ `:` type($src) `)`)?
              (`outs` `(` $dst^  `:` type($dst) `)`)?
              (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
              (`broadcast` `=` $broadcast^)?
              (`transpose` `=` $transpose^)?
              (`->` type($result)^)?
From the Elementwise Nary Vector Op template:

This operation performs element-wise operation on N operands and produces a single result. It may perform either transpose or broadcast along the way (but not both).

Common constraints:

Follows DestinationStyleOpInterface.

The number of input operands is N; the number of output/result is one.

The input/init operands and result have the same rank.

The first input is vector-only.

Additional constraints:

The input/init operands and result have the same element type.

Support only Vector-Vector operation.

Traits: , , , , , , , , , , , , AlwaysSpeculatableImplTraitAttrSizedOperandSegmentsCollapsibleConsecutiveTargetDimsTraitElementwiseNaryOpTrait<2>HIVMOpSameOperandsAndResultRankOpPipeTrait<PIPE::PIPE_V>SameOperandsElementTypeSinglePipeOpTraitTransposableOTFUniformReassociationFlattenTraitVectorCoreTypeTraitVectorOnlyTrait<0>VectorOnlyTrait<1>

Interfaces: , , , , , , , , ConditionallySpeculatableDestinationStyleOpInterfaceExtraBufferOpInterfaceFlattenInterfaceHIVMCoreTypeInterfaceHIVMStructuredOpInterfaceHIVMStructuredOpMemoryEffectsOpInterfaceOpPipeInterface

Attributes:
Attribute	MLIR Type	Description
transpose	::mlir::DenseI64ArrayAttr	i64 dense array attribute
broadcast	::mlir::DenseI64ArrayAttr	i64 dense array attribute
Operands:
Operand

Description

src

variadic of shaped of any type values

dst

variadic of shaped of any type values

temp_buffer

memref of any type values

Results:
Result

Description

result

variadic of ranked tensor of any type values

hivm.hir.wait_flag (hivm::WaitFlagOp)
Hivm wait flag.

Syntax:

operation ::= `hivm.hir.wait_flag` `[`
              $set_pipe
              `,` $wait_pipe
              `,` custom<EventID>($static_event_id, $dynamic_event_id)
              `]` attr-dict
Interfaces: InferCoreTypeInterface

Attributes:
Attribute	MLIR Type	Description
set_pipe	::mlir::hivm::PipeAttr	
wait_pipe	::mlir::hivm::PipeAttr	
static_event_id	::mlir::hivm::EventAttr	
Operands:
Operand

Description

dynamic_event_id

64-bit signless integer

Attributes
AddressSpaceAttr
Syntax:

#hivm.address_space<
  ::mlir::hivm::AddressSpace   # address_space
>
HIVM address space mapping attribute. Maps to GM, L1, L0A, L0B, L0C and UB.

Parameters:
Parameter

C++ type

Description

address_space

::mlir::hivm::AddressSpace

an enum of type AddressSpace

AlignKindAttr
alignment kind information

Syntax:

#hivm.align_kind<
  ::mlir::hivm::AlignKind   # value
>
HIVM alignment kind attribute.

Parameters:
Parameter

C++ type

Description

value

::mlir::hivm::AlignKind

an enum of type AlignKind

AllocAlignDimsAttr
Syntax: #hivm.alloc_align_dims

HIVM alloc align dims.

AllocAlignValueInByteAttr
Syntax: #hivm.alloc_align_value_in_byte

HIVM alloc align value in bytes.

AtomicKindAttr
Atomic Operation Kind for StoreOp

Syntax:

#hivm.atomic_kind<
  ::mlir::hivm::AtomicKind   # value
>
HIVM atomic store kind attribute.

Parameters:
Parameter

C++ type

Description

value

::mlir::hivm::AtomicKind

an enum of type AtomicKind

AxisKindAttr
hivm op axis kind information

Syntax:

#hivm.axis_kind<
  ::mlir::hivm::AxisKind   # value
>
HIVM op axis kind attribute.

Parameters:
Parameter

C++ type

Description

value

::mlir::hivm::AxisKind

an enum of type AxisKind

HIVMBlockMappingAttr
Syntax:

#hivm.block<
  std::optional<int32_t>   # order
>
Parameters:
Parameter

C++ type

Description

order

std::optional<int32_t>

CompareModeAttr
Compare Mode for VCmpOp

Syntax:

#hivm.compare_mode<
  ::mlir::hivm::CompareMode   # value
>
HIVM compare mode attribute.

Parameters:
Parameter

C++ type

Description

value

::mlir::hivm::CompareMode

an enum of type CompareMode

DCCIModeAttr
hivm dcci mode

Syntax:

#hivm.DCCIMode<
  ::mlir::hivm::DCCIMode   # value
>
HIVM DCCI mode attribute.

Parameters:
Parameter

C++ type

Description

value

::mlir::hivm::DCCIMode

an enum of type DCCIMode

DataCacheKindAttr
hivm data cache kind

Syntax:

#hivm.DataCacheKind<
  ::mlir::hivm::DataCacheKind   # value
>
HIVM data cache kind attribute.

Parameters:
Parameter

C++ type

Description

value

::mlir::hivm::DataCacheKind

an enum of type DataCacheKind

DataLayoutAttr
Syntax:

#hivm.data_layout<
  ::mlir::hivm::DataLayout,   # data_layout
  std::optional<bool>,   # transpose
  std::optional<DenseI64ArrayAttr>   # fractalSizes
>
HIVM data layout mapping attribute. Maps to DOTA_ND, DOTB_ND, DOTC_ND, zN, nZ and ND.

transpose: Indicates that the layout is transposed. Only valid and must be present for DOTA_ND and DOTB_ND layout.

Parameters:
Parameter

C++ type

Description

data_layout

::mlir::hivm::DataLayout

an enum of type DataLayout

transpose

std::optional<bool>

fractalSizes

std::optional<DenseI64ArrayAttr>

DeinterleaveModeAttr
HIVM deinterleave mode

Syntax:

#hivm.deinterleave_mode<
  ::mlir::hivm::DeinterleaveMode   # value
>
HIVM deinterleave index mode

Parameters:
Parameter

C++ type

Description

value

::mlir::hivm::DeinterleaveMode

an enum of type DeinterleaveMode

DescaleModeAttr
descale mode for matmul

Syntax:

#hivm.descale_mode<
  ::mlir::hivm::DescaleMode   # value
>
HIVM descale mode attribute for matmul op.

Parameters:
Parameter

C++ type

Description

value

::mlir::hivm::DescaleMode

an enum of type DescaleMode

DisableAutoInjectBlockSyncAttr
Syntax: #hivm.disable_auto_inject_block_sync

Disable auto inject block sync, skip block sync injection.

EventAttr
Syntax:

#hivm.event<
  ::mlir::hivm::EVENT   # event
>
HIVM event attribute for synchronization.

Parameters:
Parameter

C++ type

Description

event

::mlir::hivm::EVENT

an enum of type EVENT

FixpipePreQuantModeAttr
HIVM fixpipe pre_quant mode

Syntax:

#hivm.fixpipe_pre_quant_mode<
  ::mlir::hivm::FixpipePreQuantMode   # value
>
HIVM fixpipe pre_quant mode

Parameters:
Parameter

C++ type

Description

value

::mlir::hivm::FixpipePreQuantMode

an enum of type FixpipePreQuantMode

FixpipePreReluModeAttr
HIVM fixpipe pre_relu mode

Syntax:

#hivm.fixpipe_pre_relu_mode<
  ::mlir::hivm::FixpipePreReluMode   # value
>
HIVM fixpipe pre_relu mode

Parameters:
Parameter

C++ type

Description

value

::mlir::hivm::FixpipePreReluMode

an enum of type FixpipePreReluMode

HIVMFuncDynMemrefArgsAttr
Syntax: #hivm.func_dyn_memref_args

HIVM FuncDynMemrefArgs to mark the index array of dynamic memref arguments of function.

InsertSliceSourceIndexAttr
Syntax: #hivm.insert_slice_source_index

Specifies which operand is insert_slice source in vconcat op

MultiBufferAttr
Syntax: #hivm.multi_buffer

HIVM multi-buffer attribute.

PadModeAttr
Syntax:

#hivm.padmode<
  ::mlir::hivm::PadMode   # padmode
>
HIVM pad mode attribute.

Parameters:
Parameter

C++ type

Description

padmode

::mlir::hivm::PadMode

an enum of type PadMode

ParallelLoopAttr
Syntax: #hivm.parallel_loop

Specifies that marked loop can run in parallel.

PipeAttr
Syntax:

#hivm.pipe<
  ::mlir::hivm::PIPE   # pipe
>
HIVM Op pipe attribute.

Parameters:
Parameter

C++ type

Description

pipe

::mlir::hivm::PIPE

an enum of type PIPE

ReduceOpAttr
Syntax:

#hivm.reduce_op<
  ::mlir::hivm::ReduceOperation   # reduce_op
>
HIVM reduction arithmetic operation attribute.

Parameters:
Parameter

C++ type

Description

reduce_op

::mlir::hivm::ReduceOperation

an enum of type ReduceOperation

RoundModeAttr
Round Mode for VCastOp

Syntax:

#hivm.round_mode<
  ::mlir::hivm::RoundMode   # value
>
RINT: round to nearest, tie to even (c language rint)

ROUND: round to nearest, tie away from zero (c language round)

FLOOR: round to minus infinity (c language floor)

CEIL: round to positive infinity (c language ceil)

TRUNC: round to zero (c language trunc)

ODD: round to odd (Von Neumann rounding)

Parameters:
Parameter

C++ type

Description

value

::mlir::hivm::RoundMode

an enum of type RoundMode

StorageAlignedAttr
Syntax: #hivm.storage_aligned

If a module is tagged with this attribute, it means that all of the operations within all device functions in this module are aligned. If a function is tagged with this attribute, it means that all of the operations in this function are aligned.

StrideAlignDimsAttr
Syntax: #hivm.stride_align_dims

HIVM stride align dims.

StrideAlignValueInByteAttr
Syntax: #hivm.stride_align_value_in_byte

HIVM stride align value in bytes.

HIVMSubBlockMappingAttr
Syntax:

#hivm.sub_block<
  ::mlir::hivm::MappingId   # sub_block
>
HIVM sub block mapping attribute for the cv block dim ratio of mix func.

Parameters:
Parameter

C++ type

Description

sub_block

::mlir::hivm::MappingId

an enum of type MappingId

SyncBlockInstrModeAttr
Syntax:

#hivm.sync_block_instr_mode<
  ::mlir::hivm::SyncBlockInstrMode   # sync_instr_mode
>
HIVM synchronization block instruction mode attribute.

Parameters:
Parameter

C++ type

Description

sync_instr_mode

::mlir::hivm::SyncBlockInstrMode

an enum of type SyncBlockInstrMode

SyncBlockModeAttr
Syntax:

#hivm.sync_block_mode<
  ::mlir::hivm::SyncBlockMode   # sync_mode
>
HIVM synchronization block mode attribute.

Parameters:
Parameter

C++ type

Description

sync_mode

::mlir::hivm::SyncBlockMode

an enum of type SyncBlockMode

TCoreTypeAttr
Syntax:

#hivm.tcore_type<
  ::mlir::hivm::TCoreType   # tcoretype
>
HIVM op core type attribute.

Parameters:
Parameter

C++ type

Description

tcoretype

::mlir::hivm::TCoreType

an enum of type TCoreType

TCoreTypeMarkerAttr
Syntax:

#hivm.tcore_type_marker<
  ::mlir::hivm::TCoreType   # tcoretype
>
HIVM op core type marker attribute.

Parameters:
Parameter

C++ type

Description

tcoretype

::mlir::hivm::TCoreType

an enum of type TCoreType

TFuncCoreTypeAttr
Syntax:

#hivm.func_core_type<
  ::mlir::hivm::TFuncCoreType   # funcCoreType
>
HIVM function core type attribute.

Parameters:
Parameter

C++ type

Description

funcCoreType

::mlir::hivm::TFuncCoreType

an enum of type TFuncCoreType

TModuleCoreTypeAttr
Syntax:

#hivm.module_core_type<
  ::mlir::hivm::TModuleCoreType   # moduleCoreType
>
HIVM module core type attribute.

If all of the functions within the module has func core type , the module core type is .AIVAIV

If all of the functions within the module has func core type , the module core type is .AICAIC

Otherwise, the module core type is .MIX

Parameters:
Parameter

C++ type

Description

moduleCoreType

::mlir::hivm::TModuleCoreType

an enum of type TModuleCoreType

TPartOfMixAttr
Syntax: #hivm.part_of_mix

HIVM function is a part of mix kernel.

TypeFnAttr
Cast for VCastOp

Syntax:

#hivm.cast<
  ::mlir::hivm::TypeFn   # value
>
HIVM cast attribute.

Parameters:
Parameter

C++ type

Description

value

::mlir::hivm::TypeFn

an enum of type TypeFn

UnitFlagAttr
Syntax:

#hivm.unit_flag<
  ::mlir::hivm::UNIT_FLAG   # unit_flag
>
HIVM unit flag attribute for synchronization.

Parameters:
Parameter

C++ type

Description

unit_flag

::mlir::hivm::UNIT_FLAG

an enum of type UNIT_FLAG

UnlikelyConditionAttr
Syntax: #hivm.unlikely_condition

Specifies that marked condition is unlikely to evaluate to true.

VFModeAttr
HIVM VF Mode

Syntax:

#hivm.vf_mode<
  ::mlir::hivm::VFMode   # value
>
HIVM VF mode attribute.

Parameters:
Parameter

C++ type

Description

value

::mlir::hivm::VFMode

an enum of type VFMode

Enums
AddressSpace
HIVM Address Space

Cases:
Symbol

Value

String

Zero

0

zero

GM

1

gm

L1

2

cbuf

L0A

3

ca

L0B

4

cb

L0C

5

cc

UB

6

ub

AlignKind
alignment kind information

Cases:
Symbol

Value

String

ALIGN

0

align

UNALIGNED

1

unaligned

UNKNOWN

2

unknown

AtomicKind
Atomic Operation Kind for StoreOp

Cases:
Symbol

Value

String

NONE

0

none

ADD

1

add

MAX

2

max

MIN

3

min

AND

4

and

OR

5

or

XOR

6

xor

CAS

7

or

XCHG

8

xor

UMAX

9

umax

UMIN

10

umin

AxisKind
hivm op axis kind information

Cases:
Symbol

Value

String

FIRST

0

first

MIDDLE

1

middle

LAST

2

last

CompareMode
Compare Mode for VCmpOp

Cases:
Symbol

Value

String

EQ

0

eq

NE

1

ne

LT

2

lt

GT

3

gt

GE

4

ge

LE

5

le

DCCIMode
hivm dcci mode

Cases:
Symbol

Value

String

SINGLE_CACHE_LINE

0

single_cache_line

ALL_CACHE_LINES

1

all_cache_lines

DataCacheKind
hivm data cache kind

Cases:
Symbol

Value

String

ALL

0

all

UB

1

ub

OUT

2

out

ATOMIC

3

atomic

DataLayout
HIVM data layout

Cases:
Symbol

Value

String

DOTA_ND

1

dotA_ND

DOTB_ND

2

dotB_ND

DOTC_ND

3

dotC_ND

nZ

4

nZ

zN

5

zN

ND

6

ND

DeinterleaveMode
HIVM deinterleave mode

Cases:
Symbol

Value

String

CHANNEL_0

0

CHANNEL_0

CHANNEL_1

1

CHANNEL_1

ALL_CHANNELS

999

ALL_CHANNELS

DescaleMode
descale mode for matmul

Cases:
Symbol

Value

String

DescaleNull

0

DescaleNull

DescalePerChannel

1

DescalePerChannel

DescalePerTensor

2

DescalePerTensor

EVENT
Event ID for Synchronization

Cases:
Symbol

Value

String

EVENT_ID0

0

EVENT_ID0

EVENT_ID1

1

EVENT_ID1

EVENT_ID2

2

EVENT_ID2

EVENT_ID3

3

EVENT_ID3

EVENT_ID4

4

EVENT_ID4

EVENT_ID5

5

EVENT_ID5

EVENT_ID6

6

EVENT_ID6

EVENT_ID7

7

EVENT_ID7

FixpipePreQuantMode
HIVM fixpipe pre_quant mode

Cases:
Symbol

Value

String

NO_QUANT

0

NO_QUANT

S322I8

9

S322I8

F322F16

1

F322F16

F322BF16

16

F322BF16

FixpipePreReluMode
HIVM fixpipe pre_relu mode

Cases:
Symbol

Value

String

NO_RELU

0

NO_RELU

NORMAL_RELU

1

NORMAL_RELU

LEAKY_RELU

2

LEAKY_RELU

P_RELU

3

P_RELU

IteratorType
HIVM Structured Op Iterator Type

Cases:
Symbol

Value

String

kParallel

0

parallel

kBroadcast

1

broadcast

kTranspose

2

transpose

kReduction

3

reduction

kInterleave

4

interleave

kDeinterleave

5

deinterleave

kInverse

6

inverse

kPad

7

pad

kConcat

8

concat

kGather

9

gather

kCumulative

10

cumulative

kOpaque

99

opaque

MatmulBiasMode
bias mode for local matmul op

Cases:
Symbol

Value

String

NoBias

0

NoBias

PerChannelAdd

1

PerChannelAdd

PerChannelAddWithSplitK

2

PerChannelAddWithSplitK

ElementwiseCrossLoopAdd

4

ElementwiseCrossLoopAdd

ElementwiseAdd

3

ElementwiseAdd

MemPlanMode
Mem Plan Mode

Cases:
Symbol

Value

String

LOCAL_MEM_PLAN

0

LOCAL_MEM_PLAN

GLOBAL_WORKSPACE_PLAN

1

GLOBAL_WORKSPACE_PLAN

PadMode
Pad Mode for LoadOp

Cases:
Symbol

Value

String

PadNull

0

PadNull

PadFirstElem

1

PadFirstElem

PadValue

2

PadValue

PIPE
HIVM Op Pipe

Cases:
Symbol

Value

String

PIPE_S

0

PIPE_S

PIPE_V

1

PIPE_V

PIPE_M

2

PIPE_M

PIPE_MTE1

3

PIPE_MTE1

PIPE_MTE2

4

PIPE_MTE2

PIPE_MTE3

5

PIPE_MTE3

PIPE_ALL

6

PIPE_ALL

PIPE_MTE4

7

PIPE_MTE4

PIPE_MTE5

8

PIPE_MTE5

PIPE_V2

9

PIPE_V2

PIPE_FIX

10

PIPE_FIX

VIRTUAL_PIPE_MTE2_L1A

11

VIRTUAL_PIPE_MTE2_L1A

VIRTUAL_PIPE_MTE2_L1B

12

VIRTUAL_PIPE_MTE2_L1B

PIPE_NUM

13

PIPE_NUM

PIPE_UNASSIGNED

99

PIPE_UNASSIGNED

ReduceOperation
Reduction kind for VReduceOp

Cases:
Symbol

Value

String

sum

1

sum

prod

2

prod

max

3

max

min

4

min

max_with_index_left

5

max_with_index_left

max_with_index_right

6

max_with_index_right

min_with_index_left

7

min_with_index_left

min_with_index_right

8

min_with_index_right

any

9

any

all

10

all

xori

11

xori

ori

12

ori

andi

13

andi

none

0

none

RoundMode
Round Mode for VCastOp

Cases:
Symbol

Value

String

RINT

0

rint

ROUND

1

round

FLOOR

2

floor

CEIL

3

ceil

TRUNC

4

trunc

ODD

5

odd

TRUNCWITHOVERFLOW

6

truncwithoverflow

SyncBlockInstrMode
HIVM SyncBlockInstrMode

Cases:
Symbol

Value

String

INTER_BLOCK_SYNCHRONIZATION

0

INTER_BLOCK_SYNCHRONIZATION

INTER_SUBBLOCK_SYNCHRONIZATION

1

INTER_SUBBLOCK_SYNCHRONIZATION

INTRA_BLOCK_SYNCHRONIZATION

2

INTRA_BLOCK_SYNCHRONIZATION

SyncBlockMode
HIVM SyncBlockMode

Cases:
Symbol

Value

String

ALL_CUBE

0

ALL_CUBE

ALL_VECTOR

1

ALL_VECTOR

ALL_SUB_VECTOR

2

ALL_SUB_VECTOR

BARRIER_CUBE

3

BARRIER_CUBE

BARRIER_VECTOR

4

BARRIER_VECTOR

ALL

5

ALL

TCoreType
HIVM Op Core Type

Cases:
Symbol

Value

String

CUBE

1

CUBE

VECTOR

2

VECTOR

CUBE_OR_VECTOR

3

CUBE_OR_VECTOR

CUBE_AND_VECTOR

4

CUBE_AND_VECTOR

TFuncCoreType
HIVM Function Core Type

Cases:
Symbol

Value

String

AIC

1

AIC

AIV

2

AIV

MIX

3

MIX

AIC_OR_AIV

4

AIC_OR_AIV

TModuleCoreType
HIVM Module Core Type

Cases:
Symbol

Value

String

AIC

1

AIC

AIV

2

AIV

MIX

3

MIX

TypeFn
Cast for VCastOp

Cases:
Symbol

Value

String

cast_signed

0

cast_signed

cast_unsigned

1

cast_unsigned

bitcast

2

bitcast

UNIT_FLAG
Unit Flag Mode for Synchronization

Cases:
Symbol

Value

String

DISABLED

0

DISABLED

保留

1

保留

ENABLED_WITHOUT_UPDATE

2

ENABLED_WITHOUT_UPDATE

ENABLED_WITH_UPDATE

3

ENABLED_WITH_UPDATE

ENABLED_ONLY_LAST_ITER

4

ENABLED_ONLY_LAST_ITER

ENABLED_ONLY_FIRST_ITER

5

ENABLED_ONLY_FIRST_ITER

ENABLED_ONLY_FIRST_AND_LAST_ITERS

6

ENABLED_ONLY_FIRST_AND_LAST_ITERS

VFMode
HIVM VF 模式

案例：
象征

价值

弦

SIMD

0

SIMD

SIMT

1

SIMT

MIX

2

MIX

映射识别
环路映射的映射 ID

案例：
象征

价值

弦

DimX

0

x

