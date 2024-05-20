 
from __future__ import print_function
from six.moves import xrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


training_data = datasets.CIFAR10(root="data", train=True, download=True,
                                  transform=transforms.Compose([
                                   transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
                                  ]))

validation_data = datasets.CIFAR10(root="data", train=False, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                  ]))
class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)
class Decoder(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()

        # self._conv_1 = nn.Conv2d(in_channels=in_channels,
        #                          out_channels=num_hiddens,
        #                          kernel_size=3,
        #                          stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens // 2,
                                                kernel_size=4,
                                                stride=2, padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2,
                                                out_channels=3,
                                                kernel_size=4,
                                                stride=2, padding=1)

    def forward(self, x):
        # x = self._conv_1(x)
        x = self._residual_stack(x)
        x = self._conv_trans_1(x)
        x = F.relu(x)

        return self._conv_trans_2(x)

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay ,epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._mapping_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        self._mapping_conv_1 = nn.Conv2d(in_channels=embedding_dim,
                                         out_channels=num_hiddens,
                                         kernel_size=3,stride=1,padding=1)
        # self._decoder = Decoder(num_hiddens,
        #                         num_residual_layers,
        #                         num_residual_hiddens)
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs= self._mapping_conv(inputs)
        input = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = input.shape

        # Flatten input
        flat_input = input.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        # encoding_indices = torch.argmin(distances, dim=1)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=input.device)
        # print(encodings.shape)
        encodings.scatter_(1, encoding_indices, 1)
        # print(encodings.shape)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        # print(quantized.shape)
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), input)
        q_latent_loss = F.mse_loss(quantized, input.detach())
        loss = self._commitment_cost * e_latent_loss + q_latent_loss
        # Straight Through Estimator
        quantized = input + (quantized - input).detach()
        avg_probs = torch.mean(encodings, dim=0)
        # print(avg_probs)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        # kl_loss = F.kl_div(F.softmax(input.reshape(64, -1), dim=1), F.softmax(quantized.reshape(64, -1), dim=1),
        #                    reduction='mean')
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        new_vec = self._mapping_conv_1(quantized)
        # reconstruc = self._decoder(new_vec)
        # convert quantized from BHWC -> BCHW
        return loss, quantized,new_vec,perplexity, encoding_indices
# As an example for a `BCHW` tensor of shape `[16, 64, 32, 32]`, we will first convert it to an `BHWC` tensor of shape `[16, 32, 32, 64]` and then reshape it into `[16384, 64]` and all `16384` vectors of size `64`  will be quantized independently. In otherwords, the channels are used as the space in which to quantize. All other dimensions will be flattened and be seen as different examples to quantize, `16384` in this case.

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens // 2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        x = self._conv_2(x)
        x = F.relu(x)
        x = self._conv_3(x)
        x = self._residual_stack(x)
        return x

batch_size = 64
num_training_updates =20000
num_hiddens =128
num_residual_hiddens = 32
num_residual_layers = 2

# embedding_dim = 16   #D
# num_embeddings =4096    #N
commitment_cost = 0.25

decay = 0.99
n_factors= 1
learning_rate = 1e-4
cycle = 10

# def count_codebook_usage(encoding_indices,embeddings):
#     usage = [0] * embeddings
#     for index in encoding_indices:
#         usage[index] += 1
#     used_codewords = sum(1 for u in usage if u > 0)
# #     return  used_codewords
# sizes =  [512,1024, 2048, 4096, 8192, 16384, 32768,65536]    # 每个record_vector的大小
# record_vectors = []

# for size in sizes:
#     record_vector = [0] * size
#     record_vectors.append(record_vector)
class Model(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                commitment_cost, decay=0):
        super(Model, self).__init__()
        self.num_embeddings_list=[512,1024,2048,4096,8192,16384,32768,65536]
        self.embedding_dim_list = [128,64,32,16,8,4,2,1]
        self.quantization_keys = torch.nn.Parameter(torch.randn(8, 1, 128))
        self.quantization_attention = torch.nn.MultiheadAttention(embed_dim=128, num_heads=2)
        self._encoder = Encoder(3, num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens).to(device)
        self._vq_vae = nn.ModuleList(
            [VectorQuantizerEMA(self.num_embeddings_list[i], self.embedding_dim_list[i],
                                              commitment_cost, decay).to(device) for i in range(8)])
        self._decoder = Decoder(num_hiddens,
                                num_residual_layers,
                                num_residual_hiddens).to(device)
        self.register_buffer('temperature', torch.tensor(1.0))
    def forward(self, x):
        if self.training:
            Temperature=self.temperature
        else:
            Temperature=1
        h = self._encoder(x)
        query = h.view(-1, 128)
        query = query.unsqueeze(1)
        # codebook_sizes = torch.tensor(range(len(self.num_embeddings_list)), dtype=torch.long, device=x.device)
        # codebook_size_embedding = self.codebook_size_embeddings(codebook_sizes)
        # 将码本大小的嵌入表示传递给注意力键
        # quantization_keys = self.quantization_keys(codebook_size_embedding).unsqueeze(1)
        _, att_scores = self.quantization_attention(query=query, key=self.quantization_keys,
                                                value=self.quantization_keys)
        att_scores = nn.functional.gumbel_softmax(att_scores, hard=True, dim=2, tau=Temperature)
        # importance_weights_expanded = self.importance_weights.view(1, 1, -1)
        # att_scores_weighted = att_scores * importance_weights_expanded
        # print(att_scores.shape)
        vqloss = [0,0,0,0,0,0,0,0]
        used_codewords = []
        fuzadu = []
        Zs= []
        CBloss = 0
        # new_vec = [_ for _ in range(7)]
        # codeword = [0,0,0,0,0,0,0,0]
        for i in range(8):
            vqloss[i],quantized,new_vec, perixity,encoding_indices = self._vq_vae[i](h)
            # codewords = count_codebook_usage(encoding_indices,self.num_embeddings_list[i])
            # used_codewords.append(codewords)
            # for index in encoding_indices:
            #     if record_vectors[i][index] == 0:
            #         record_vectors[i][index] = 1 
            # codeword[i] = sum(record_vectors[i])
            # fuzadu.append(perixity)
            Zs.append(new_vec.reshape(-1, 128).unsqueeze(1))
            # codeword.append(encoding_indices)
            CBloss += vqloss[i]
        # 结果形状为 [7, 64, 128, 7, 7]
        ExtraLoss=CBloss/8
        Zs = torch.cat(Zs, 1)
        selected_indices = att_scores.argmax(dim=2).squeeze()
        # indices, counts = torch.unique(selected_indices, return_counts=True)
        # counts_tensor =torch.tensor(counts,device=device)
        # vq_tensor = torch.tensor(vqloss,device=device)
        # product = torch.mul(counts_tensor, vq_tensor)
        # result = torch.sum(product)
        # result /= 4096
        state = torch.bmm(att_scores.permute(1, 0, 2), Zs)
        # print(state.shape)
        state = state.squeeze(1)
        state = state.reshape(64, 128, 8, 8)
        x_recon = self._decoder(state)
        recon_loss = F.mse_loss(x, x_recon)
        # print(recon_loss)
        loss =recon_loss+ExtraLoss
        grad_h = torch.autograd.grad(recon_loss, h, retain_graph=True)
        # grad_state = torch.autograd.grad(loss, state, retain_graph=True)
        # delta_gap = torch.abs(grad_h[0] - grad_state[0])
        return x_recon, loss,recon_loss,vqloss,grad_h,h
# We use the hyperparameters from the author's code:

training_loader = DataLoader(training_data,
                             batch_size=batch_size,
                             shuffle=True,
                             pin_memory=True)

validation_loader = DataLoader(validation_data,
                               batch_size=batch_size,
                               shuffle=False,
                               pin_memory=True)

model=Model(num_hiddens, num_residual_layers, num_residual_hiddens,
                     commitment_cost, decay).to(device)
optimizer=optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
# valid_loss = []
vq_loss= []
# 使用PyTorch初始化一个形状为 [10000, 8] 的Tensor来记录索引使用次数
index_usage_counts = torch.zeros((num_training_updates, 8), dtype=torch.long).to(device)
codebook_loss =  torch.zeros((num_training_updates, 8), dtype=torch.float).to(device)
# used_codewords =  torch.zeros((num_training_updates, 8), dtype=torch.long).to(device)
# xuexidu = torch.zeros((num_training_updates, 8), dtype=torch.float).to(device)
# each_used_codewords =  torch.zeros((num_training_updates, 8), dtype=torch.long).to(device)
tidugap = []
model.train()
for m in xrange(num_training_updates):
    model.temperature = torch.tensor((num_training_updates - m) * 1 + 1, dtype=torch.float32, device=device)
    # model.temperature = 5
    (data, _) = next(iter(training_loader))
    # data = data.repeat(1, 3, 1, 1)
    data = data.to(device)
    optimizer.zero_grad()
    data_recon, loss,recon_loss,vqloss,grad_h,h= model(data)
    
    # index_usage_counts[m, indices] += counts
    # codebook_loss[m, :] = torch.tensor(vqloss)
    # used_codewords[m, :] = torch.tensor(codeword)
    # xuexidu[m, :] = torch.tensor(fuzadu)
    # each_used_codewords[m, :] = torch.tensor(unique_counts)
    # loss.requires_grad_(True)
    # train_loss.append(recon_loss.item())
    # vq_loss.append(result.item())
    # zq = model._decoder(h)
    # reconl = F.mse_loss(zq,data)
    # grad_state = torch.autograd.grad(reconl, h, retain_graph=True)
    # reconl.backward()
    # optimizer.step()
    loss.backward()
    optimizer.step()
    zq = model._decoder(h)
    reconl = F.mse_loss(zq,data)
    grad_state = torch.autograd.grad(reconl, h, retain_graph=True)
    delta_gap = torch.abs(grad_h[0] - grad_state[0])
    mean_delta_gap = torch.mean(delta_gap)
    tidugap.append(mean_delta_gap.item())
    # (valid, _) = next(iter(validation_loader))
    # # valid_original = valid_original.repeat(1, 3, 1, 1)
    # valid = valid.to(device)
    # _,_,rloss,_,_,_,_ = model(valid)
    # valid_loss.append(rloss.item())
    if (m + 1) % 100 == 0:
        # print(indices)
        # print(counts)
        # print(model.quantization_keys)
        # print(model.importance_weights)
        print('%d iterations' % (m + 1))
        print()
# torch.save(vq_loss,'gumbel_cifar_lianghualoss.pth')
# 将记录Tensor保存到硬盘
# torch.save(index_usage_counts, 'index_usage_counts_cifar_20000_7.pth')
# torch.save(codebook_loss, 'codebook_loss_cifar_gumbel.pth')
# torch.save(tidugap, 'tidugap_cifar_20000.pth')
# torch.save(num_used_codewords, 'num_used_codewords_cifar_20000_2.pth')
# torch.save(xuexidu, 'xuexidu_cifar_20000_6.pth')
# torch.save(used_codewords, 'used_codewords_cifar_20000_3.pth')
# torch.save(each_used_codewords, 'each_used_codewords_cifar_20000_2.pth')
# mse_loss = 0
# torch.save(train_loss, 'train_loss_cifar_20000_5.pth')
# torch.save(valid_loss, 'valid_loss_cifar_20000_5.pth')
mse_loss = []
model.eval()
for i in range(156):
    (valid_original, _) = next(iter(validation_loader))
    # valid_original = valid_original.repeat(1, 3, 1, 1)
    valid_original = valid_original.to(device)
    valid_r,_,recon_loss,_,_,_ = model(valid_original)
    mse_loss.append(recon_loss.item())
# print(mse_loss)
torch.save(mse_loss,'mseloss_cifar_65536_gumbel_2.pth')
