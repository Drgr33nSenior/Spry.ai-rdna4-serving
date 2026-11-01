"""Prospective synthetic sixteen-row reference; no framework imported here.

FP32 matmul/BF16 output are preserved. M16 instead of full-M BLAS can change
reduction order. Exact equality is still mandatory; no tolerance is introduced.
"""


def quantizer_geometry(m,k):
    if type(m)is not int or not 1<=m<=2048 or type(k)is not int or k not in(3072,5120,8704):raise ValueError('bounded private adapter M1..2048 geometry')
    if m<=32:
        block_m=1<<(m-1).bit_length();block_n=32;iterations=1;warps=1;stages=1
    else:
        block_m=32;block_n=128;iterations=4;warps=4;stages=2
    return dict(grid=((m+block_m-1)//block_m,(k+block_n*iterations-1)//(block_n*iterations)),BLOCK_SIZE_M=block_m,BLOCK_SIZE_N=block_n,NUM_ITER=iterations,NUM_STAGES=stages,num_warps=warps,waves_per_eu=0,num_stages=1)


def column_tiles(n):
    if type(n)is not int or n not in(17408,5120,8192,7168,48):raise ValueError('declared fused output width')
    return tuple((begin,min(begin+256,n))for begin in range(0,n,256))


def reference_basis(packed,scales,weight,weight_scale,*,torch,dequantize,check,deadline):
    """Caller supplies exact v2 scalar format/dequantizer and source-bound tensors.

    Calculate this before allocating full-M input/output/graph buffers. Only the
    returned BF16 basis survives; all FP32 reference temporaries are local.
    """
    if len(packed.shape)!=2 or packed.shape[0]!=16 or tuple(scales.shape)!=(16,packed.shape[1]//16):raise ValueError('sixteen complete quantized input rows')
    n,k=weight.shape[0],weight.shape[1]*2
    quantizer_geometry(16,k)
    if packed.shape[1]*2!=k or tuple(weight_scale.shape)!=(n,k//32):raise ValueError('reference packed/scales dimensions')
    tiles=column_tiles(n);check(deadline)
    x_float=dequantize(packed,scales,torch)
    output=torch.empty((16,n),dtype=torch.bfloat16,device=packed.device)
    for begin,end in tiles:
        check(deadline)
        tile=dequantize(weight[begin:end],weight_scale[begin:end],torch)
        output[:,begin:end]=(x_float@tile.T).to(torch.bfloat16)
        del tile
    return output
