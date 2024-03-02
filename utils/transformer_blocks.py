import torch

def attention_layer(q_mat:torch.Tensor,
                    k_mat:torch.Tensor,
                    v_mat:torch.Tensor)->torch.Tensor:
    """ attention_layer implementation.
    ### Args:
        Q (torch.Tensor): Query tensor (Tin x dk).
        K (torch.Tensor): Key tensor (Tout x dk).
        V (torch.Tensor): Value tensor (Tout x dv).
    ### Returns:
        torch.Tensor: Attention tensor.
    """
    scores = torch.matmul(q_mat,k_mat.t())/torch.sqrt(k_mat.shape[-1])
    scores = torch.softmax(scores,dim=-1)
    return torch.matmul(scores,v_mat)

def transformer_block(x:torch.Tensor,
                      ANN:torch.nn.Module)->torch.Tensor:
    """ transformer_block implementation.
    ### Args:
        x (torch.Tensor): in vector
        ANN (torch.nn.Module): Artificial neural network

    ### Returns:
        torch.Tensor: out vector
    """
    x += positional_encoding(x.shape)
    x = torch.nn.LayerNorm(x+torch.nn.MultiheadAttention(x))
    x = torch.nn.LayerNorm(x+ANN(x))
    return x

def positional_encoding(in_shape:torch.Size)->torch.Tensor:
    """ positional_encoding implementation.

    ### Args:
        in_shape (torch.Size): input shape that needs to be encoded

    ### Returns:
        torch.Tensor: the PE tensor representing the positional encoding
    """
    T,d = in_shape
    pe = torch.zeros(in_shape)
    for pos in range(T):
        for i in range(d):
            if i%2 is 0:
                pe[pos,i] = torch.sin(pos/10000**(i/d))
            else:
                pe[pos,i] = torch.cos(pos/10000**(i/d))
    return pe

# def multi_head_attention_layer(x_mat:torch.Tensor,
#                                wo_mat:torch.Tensor,
#                                q_mat_batch:torch.Tensor,
#                                k_mat_batch:torch.Tensor,
#                                v_mat_batch:torch.Tensor)->torch.Tensor:
    
    
#     """# multi_head_attention_layer
#     Multi-head attention layer.
#     ### Args:
#         x_mat (torch.Tensor): Input tensor (Tin x dv).
#         wo_mat (torch.Tensor): Output tensor (Tout x dv).
#     ### Returns:
#         torch.Tensor: Attention tensor.
#     """
