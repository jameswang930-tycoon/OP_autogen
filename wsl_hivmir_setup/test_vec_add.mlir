// 测试文件: Vector Add in HIVM Dialect
// 用于验证 bishengir-compile / bishengir-opt 工具链
// 用法: bishengir-compile test_vec_add.mlir -o vec_add.o
//      bishengir-opt test_vec_add.mlir -convert-hfusion-to-hivm --print-ir-after-all 2>&1 | tee hivm_dump.log

func.func @add_kernel(%arg0: memref<1024xf16, #hivm.address_space<gm>>,
                       %arg1: memref<1024xf16, #hivm.address_space<gm>>,
                       %arg2: memref<1024xf16, #hivm.address_space<gm>>)
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {

    // alloc UB buffers
    %buf_a = memref.alloc() : memref<1024xf16, #hivm.address_space<ub>>
    %buf_b = memref.alloc() : memref<1024xf16, #hivm.address_space<ub>>
    %buf_c = memref.alloc() : memref<1024xf16, #hivm.address_space<ub>>

    // DMA: GM → UB (MTE2 流水线)
    hivm.hir.load ins(%arg0 : memref<1024xf16, #hivm.address_space<gm>>)
                 outs(%buf_a : memref<1024xf16, #hivm.address_space<ub>>)
    hivm.hir.load ins(%arg1 : memref<1024xf16, #hivm.address_space<gm>>)
                 outs(%buf_b : memref<1024xf16, #hivm.address_space<ub>>)

    // Vector compute: vadd (VecUnit 流水线)
    hivm.hir.vadd ins(%buf_a, %buf_b : memref<1024xf16, #hivm.address_space<ub>>,
                                       memref<1024xf16, #hivm.address_space<ub>>)
                 outs(%buf_c : memref<1024xf16, #hivm.address_space<ub>>)

    // DMA: UB → GM (MTE3 流水线)
    hivm.hir.store ins(%buf_c : memref<1024xf16, #hivm.address_space<ub>>)
                  outs(%arg2 : memref<1024xf16, #hivm.address_space<gm>>)

    return
}
