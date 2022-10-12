import torch
import torch.nn as nn



class CResnetBlockConv1d(nn.Module):
    ''' Conditional batch normalization-based Resnet block class.

    Args:
        c_dim (int): dimension of latend conditioned code c
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
        norm_method (str): normalization method
        legacy (bool): whether to use legacy blocks 
    '''

    def __init__(self, c_dim, size_in, size_h=None, size_out=None,
                 norm_method='batch_norm'):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out

        # Submodules
        self.bn_0 = CBatchNorm1d(
            c_dim, size_h, norm_method=norm_method)
        self.bn_1 = CBatchNorm1d(
            c_dim, size_h, norm_method=norm_method)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = nn.Sigmoid()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c):
        x_z = self.fc_0(x)
        z = self.actvn(self.bn_0(x_z, c))
        z = self.fc_1(z)
        z = self.bn_1(z, c)
        z = z + x_z
        out = self.actvn(z)
        return out


        # net = self.fc_0(self.actvn(self.bn_0(x, c)))
        # dx = self.fc_1(self.actvn(self.bn_1(net, c)))

        # if self.shortcut is not None:
        #     x_s = self.shortcut(x)
        # else:
        #     x_s = x

        # return x_s + dx





class CBatchNorm1d(nn.Module):
    ''' Conditional batch normalization legacy layer class.

    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
        norm_method (str): normalization method
    '''

    def __init__(self, c_dim, f_dim, norm_method='batch_norm'):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm_method = norm_method
        # Submodules
        self.fc_gamma = nn.Linear(c_dim, f_dim)
        self.fc_beta = nn.Linear(c_dim, f_dim)
        if norm_method == 'batch_norm':
            self.bn = nn.BatchNorm1d(f_dim, affine=False)
        elif norm_method == 'instance_norm':
            self.bn = nn.InstanceNorm1d(f_dim, affine=False)
        elif norm_method == 'group_norm':
            self.bn = nn.GroupNorm(num_groups=1, num_channels=f_dim, affine=False)
        else:
            raise ValueError('Invalid normalization method!')
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.fc_gamma.weight)
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.ones_(self.fc_gamma.bias)
        nn.init.zeros_(self.fc_beta.bias)

    def forward(self, x, c):
        batch_size = x.size(0)
        # Affine mapping
        gamma = self.fc_gamma(c)
        beta = self.fc_beta(c)
        gamma = gamma.view(batch_size, self.f_dim, 1)
        beta = beta.view(batch_size, self.f_dim, 1)
        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta

        return out


# class CResnetBlockConv1d(nn.Module):
#     ''' Conditional batch normalization-based Resnet block class.

#     Args:
#         c_dim (int): dimension of latend conditioned code c
#         size_in (int): input dimension
#         size_out (int): output dimension
#         size_h (int): hidden dimension
#         norm_method (str): normalization method
#         legacy (bool): whether to use legacy blocks 
#     '''

#     def __init__(self, c_dim, size_in, size_h=None, size_out=None,
#                  norm_method='batch_norm'):
#         super().__init__()
#         # Attributes
#         if size_h is None:
#             size_h = size_in
#         if size_out is None:
#             size_out = size_in

#         self.size_in = size_in
#         self.size_h = size_h
#         self.size_out = size_out

#         # Submodules
#         self.bn_0 = CBatchNorm1d(
#             c_dim, size_in, norm_method=norm_method)
#         self.bn_1 = CBatchNorm1d(
#             c_dim, size_h, norm_method=norm_method)

#         self.fc_0 = nn.Conv1d(size_in, size_h, 1)
#         self.fc_1 = nn.Conv1d(size_h, size_out, 1)
#         self.actvn = nn.ReLU()

#         if size_in == size_out:
#             self.shortcut = None
#         else:
#             self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)
#         # Initialization
#         nn.init.zeros_(self.fc_1.weight)

#     def forward(self, x, c):
#         net = self.fc_0(self.actvn(self.bn_0(x, c)))
#         dx = self.fc_1(self.actvn(self.bn_1(net, c)))

#         if self.shortcut is not None:
#             x_s = self.shortcut(x)
#         else:
#             x_s = x

#         return x_s + dx