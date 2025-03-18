# coding: utf-8

import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender


class LightGCN(GeneralRecommender):
    def __init__(self, config, dataset):
        super(LightGCN, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']

        self.lambda_coeff = config['lambda_coeff']
        self.cf_model = config['cf_model']
        
        self.knn_k = config['knn_k']
        self.n_layers = config['n_mm_layers']

        self.n_ui_layers = config['n_ui_layers']
        self.reg_weight = config['reg_weight']
        self.kl_weight = config['kl_weight']
        self.neighbor_weight = config['neighbor_weight']
        self.build_item_graph = True

        self.n_nodes = self.n_users + self.n_items

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_adj_mat()
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=True)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=True)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
        if self.f_feat is not None:
            self.mm_embedding = nn.Embedding.from_pretrained(self.f_feat, freeze=False)
            self.mm_ln = nn.LayerNorm(self.f_feat.shape[1])
            self.mm_trs = nn.Linear(self.f_feat.shape[1], self.embedding_dim)
        self.gate_mm = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim),
                                    nn.Sigmoid())

        # self.image_adj, self.text_adj, self.mm_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach(), self.text_embedding.weight.detach(), self.mm_embedding.weight.detach())
        self.text_adj, self.image_adj, self.mm_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach(), self.image_embedding.weight.detach(), self.mm_embedding.weight.detach())

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        self.item_graph_dict = np.load(os.path.join(dataset_path, config['item_graph_dict_file']),
                                       allow_pickle=True).item()
        
        __, self.session_adj = self.get_session_adj()

    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            # norm_adj = adj.dot(d_mat_inv)
            # print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)


    def get_knn_adj_mat(self, v_embeddings, t_embeddings, f_embeddings):
        v_context_norm = v_embeddings.div(torch.norm(v_embeddings, p=2, dim=-1, keepdim=True))
        v_sim = torch.mm(v_context_norm, v_context_norm.transpose(1, 0))

        t_context_norm = t_embeddings.div(torch.norm(t_embeddings, p=2, dim=-1, keepdim=True))
        t_sim = torch.mm(t_context_norm, t_context_norm.transpose(1, 0))

        f_context_norm = f_embeddings.div(torch.norm(f_embeddings, p=2, dim=-1, keepdim=True))
        f_sim = torch.mm(f_context_norm, f_context_norm.transpose(1, 0))

        mask_v = v_sim < v_sim.mean()
        mask_t = t_sim < t_sim.mean()
        mask_f = f_sim < f_sim.mean()

        t_sim[mask_v] = 0
        v_sim[mask_t] = 0
        t_sim[mask_f] = 0
        v_sim[mask_f] = 0
        f_sim[mask_v] = 0
        f_sim[mask_t] = 0
        f_sim[mask_f] = 0
        t_sim[mask_t] = 0
        v_sim[mask_v] = 0

        index_x = []
        index_v = []
        index_t = []
        index_f = []

        all_items = np.arange(self.n_items).tolist()

        def _random():
            rd_id = random.sample(all_items, 9)  # [0]
            return rd_id

        for i in range(self.n_items):
            item_num = len(torch.nonzero(t_sim[i]))
            if item_num <= self.knn_k:
                _, v_knn_ind = torch.topk(v_sim[i], item_num)
                _, t_knn_ind = torch.topk(t_sim[i], item_num)
                _, f_knn_ind = torch.topk(f_sim[i], item_num)
            else:
                _, v_knn_ind = torch.topk(v_sim[i], self.knn_k)
                _, t_knn_ind = torch.topk(t_sim[i], self.knn_k)
                _, f_knn_ind = torch.topk(f_sim[i], self.knn_k)

            index_x.append(torch.ones_like(v_knn_ind) * i)
            index_v.append(v_knn_ind)
            index_t.append(t_knn_ind)
            index_f.append(f_knn_ind)

        index_x = torch.cat(index_x, dim=0).cuda()
        index_v = torch.cat(index_v, dim=0).cuda()
        index_t = torch.cat(index_t, dim=0).cuda()
        index_f = torch.cat(index_f, dim=0).cuda()

        adj_size = (self.n_items, self.n_items)
        del v_sim, t_sim, f_sim

        v_indices = torch.stack((torch.flatten(index_x), torch.flatten(index_v)), 0)
        t_indices = torch.stack((torch.flatten(index_x), torch.flatten(index_t)), 0)
        f_indices = torch.stack((torch.flatten(index_x), torch.flatten(index_f)), 0)
        # norm
        return self.compute_normalized_laplacian(v_indices, adj_size), self.compute_normalized_laplacian(t_indices,
                                                                                                         adj_size), self.compute_normalized_laplacian(f_indices, adj_size)


    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def get_session_adj(self):
        index_x = []
        index_y = []
        values = []
        for i in range(self.n_items):
            index_x.append(i)
            index_y.append(i)
            values.append(1)
            if i in self.item_graph_dict.keys():
                item_graph_sample = self.item_graph_dict[i][0]
                item_graph_weight = self.item_graph_dict[i][1]

                for j in range(len(item_graph_sample)):
                    index_x.append(i)
                    index_y.append(item_graph_sample[j])
                    values.append(item_graph_weight[j])
        index_x = torch.tensor(index_x, dtype=torch.long)
        index_y = torch.tensor(index_y, dtype=torch.long)
        indices = torch.stack((index_x, index_y), 0).to(self.device)
        # norm
        return indices, self.compute_normalized_laplacian(indices, (self.n_items, self.n_items))

    '''
    def generate_pesudo_labels(self, prob1, prob2, prob3, prob4):
        positive = prob1 + prob2 + prob3 + prob4 + prob4
        _, mm_pos_ind = torch.topk(positive, 10, dim=-1)
        prob = prob4.clone()
        prob.scatter_(1, mm_pos_ind, 0)
        _, single_pos_ind = torch.topk(prob, 10, dim=-1)
        return mm_pos_ind, single_pos_ind
    '''

    def generate_pesudo_labels(self, prob1, prob2, prob3):
        positive = prob1 + prob2 + prob3 + prob3
        _, mm_pos_ind = torch.topk(positive, 10, dim=-1)
        prob = prob3.clone()
        prob.scatter_(1, mm_pos_ind, 0)
        _, single_pos_ind = torch.topk(prob, 10, dim=-1)
        return mm_pos_ind, single_pos_ind
        

    def neighbor_discrimination(self, mm_positive, s_positive, emb, aug_emb, temperature=0.2):
        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), dim=2)

        n_aug_emb = F.normalize(aug_emb, dim=1)
        n_emb = F.normalize(emb , dim=1)

        mm_pos_emb = n_aug_emb[mm_positive]
        s_pos_emb = n_aug_emb[s_positive]

        emb2 = torch.reshape(n_emb, [-1, 1, self.embedding_dim])
        emb2 = torch.tile(emb2, [1, 10, 1])
        
        mm_pos_score = score(emb2, mm_pos_emb)
        s_pos_score = score(emb2, s_pos_emb)
        ttl_score = torch.matmul(n_emb, n_aug_emb.transpose(0, 1))

        mm_pos_score = torch.sum(torch.exp(mm_pos_score / temperature), dim=1)
        s_pos_score = torch.sum(torch.exp(s_pos_score / temperature), dim=1)
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1) # 1

        cl_loss = - torch.log(mm_pos_score / (ttl_score) + 10e-10) - torch.log(s_pos_score / (ttl_score - mm_pos_score) + 10e-10)
        return torch.mean(cl_loss)

    def KL(self, p1, p2):
        return p1 * torch.log(p1) - p1 * torch.log(p2) + \
               (1 - p1) * torch.log(1 - p1) - (1 - p1) * torch.log(1 - p2)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        '''
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A.update(data_dict)
        '''

        user_item_interactions = zip(inter_M.row, inter_M.col + self.n_users)
        item_user_interactions = zip(inter_M_t.row + self.n_users, inter_M_t.col)
        for u, i in user_item_interactions:
            A[u, i] = 1

        for i, u in item_user_interactions:
            A[i, u] = 1
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    '''
    def calculate_modal_weights(self, h_t, h_v, h_s, h_f):
        """
        计算三个模态的权重，基于每个模态特征向量的稀疏度和信息密度。

        参数:
            h_t (torch.Tensor): 文本模态特征向量，形状为 [N, D]，其中 N 是物品数量，D 是嵌入维度
            h_v (torch.Tensor): 图像模态特征向量，形状为 [N, D]
            h_s (torch.Tensor): 会话模态特征向量，形状为 [N, D]

        返回:
            weight_t (torch.Tensor): 文本模态的权重
            weight_v (torch.Tensor): 图像模态的权重
            weight_s (torch.Tensor): 会话模态的权重
        """

        # 计算稀疏度，可以使用 L1 范数来衡量特征的稀疏性
        sparsity_t = torch.norm(h_t, p=1, dim=-1).mean()  # 文本模态的平均稀疏度
        sparsity_v = torch.norm(h_v, p=1, dim=-1).mean()  # 图像模态的平均稀疏度
        sparsity_s = torch.norm(h_s, p=1, dim=-1).mean()  # 会话模态的平均稀疏度
        sparsity_f = torch.norm(h_f, p=1, dim=-1).mean()  # 会话模态的平均稀疏度

        # 根据稀疏度计算信息密度（稀疏度越低，密度越高）
        density_t = 1.0 / (sparsity_t + 1e-8)
        density_v = 1.0 / (sparsity_v + 1e-8)
        density_s = 1.0 / (sparsity_s + 1e-8)
        density_f = 1.0 / (sparsity_f + 1e-8)

        # 归一化信息密度，将它们转化为权重
        total_density = density_t + density_v + density_s + density_f
        weight_t = density_t / total_density
        weight_v = density_v / total_density
        weight_s = density_s / total_density
        weight_f = density_f / total_density
        
        return weight_t, weight_v, weight_s, weight_f
    '''
    def calculate_modal_weights(self, h_t, h_v, h_s):
        """
        计算三个模态的权重，基于每个模态特征向量的稀疏度和信息密度。

        参数:
            h_t (torch.Tensor): 文本模态特征向量，形状为 [N, D]，其中 N 是物品数量，D 是嵌入维度
            h_v (torch.Tensor): 图像模态特征向量，形状为 [N, D]
            h_s (torch.Tensor): 会话模态特征向量，形状为 [N, D]

        返回:
            weight_t (torch.Tensor): 文本模态的权重
            weight_v (torch.Tensor): 图像模态的权重
            weight_s (torch.Tensor): 会话模态的权重
        """

        # 计算稀疏度，可以使用 L1 范数来衡量特征的稀疏性
        sparsity_t = torch.norm(h_t, p=1, dim=-1).mean()  # 文本模态的平均稀疏度
        sparsity_v = torch.norm(h_v, p=1, dim=-1).mean()  # 图像模态的平均稀疏度
        sparsity_s = torch.norm(h_s, p=1, dim=-1).mean()  # 会话模态的平均稀疏度


        # 根据稀疏度计算信息密度（稀疏度越低，密度越高）
        density_t = 1.0 / (sparsity_t + 1e-8)
        density_v = 1.0 / (sparsity_v + 1e-8)
        density_s = 1.0 / (sparsity_s + 1e-8)


        # 归一化信息密度，将它们转化为权重
        total_density = density_t + density_v + density_s
        weight_t = density_t / total_density
        weight_v = density_v / total_density
        weight_s = density_s / total_density


        return weight_t, weight_v, weight_s

    def label_prediction(self, emb, aug_emb):
        n_emb = F.normalize(emb, dim=1)
        n_aug_emb = F.normalize(aug_emb, dim=1)
        prob = torch.mm(n_emb, n_aug_emb.transpose(0, 1))
        prob = F.softmax(prob, dim=1)
        del n_emb, n_aug_emb
        return prob

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def forward(self, IsTrain):
        mm_feats = self.mm_trs(self.mm_ln(self.mm_embedding.weight))
        mm_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_mm(mm_feats))
        for i in range(self.n_layers):
            mm_item_embeds = torch.sparse.mm(self.mm_adj, mm_item_embeds)

        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        content_embeds = all_embeddings

        mm_user_embeds = torch.sparse.mm(self.R, mm_item_embeds)

        mm_embeds = torch.cat([mm_user_embeds, mm_item_embeds], dim=0)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        del ego_embeddings, side_embeddings


        # text emb
        h_t = self.item_id_embedding.weight.clone()
        for i in range(self.n_layers):
            h_t = torch.sparse.mm(self.text_adj, h_t)

        # image emb
        h_v = self.item_id_embedding.weight.clone()
        for i in range(self.n_layers):
            h_v = torch.sparse.mm(self.image_adj, h_v)

        # session emb
        h_s = self.item_id_embedding.weight.clone()
        for i in range(self.n_layers):
            h_s = torch.sparse.mm(self.session_adj, h_s)

        if IsTrain:
            return u_g_embeddings, i_g_embeddings, h_v, h_t, h_s, mm_embeds, content_embeds

        else:
            return u_g_embeddings, i_g_embeddings, h_v, h_t, h_s


    def add_noise_to_modalities(self, h_t, h_v, h_f, noise_ratio):
        """
        根据50%的概率对文本和图像模态中的一个进行加噪声。

        参数:
        h_t (torch.Tensor): 文本模态的嵌入向量，形状为 [N, D]
        h_v (torch.Tensor): 图像模态的嵌入向量，形状为 [N, D]
        noise_ratio (float): 加噪声的比例

        返回:
        torch.Tensor, torch.Tensor: 加噪声后的文本和图像模态嵌入向量
        """
        # 随机选择哪个模态加噪声（50%概率）
        if 0 > 0.5:
            h_t, h_f = self.add_noise_to_embedding(h_t, h_f, noise_ratio)  # 对文本模态加噪声
        else:
            h_v, h_f = self.add_noise_to_embedding(h_v, h_f, noise_ratio)  # 对图像模态加噪声

        return h_t, h_v, h_f

    def add_noise_to_embedding(self, h, h_f, noise_ratio):
        """
        对单一模态的嵌入进行噪声添加，按物品个数来进行。

        参数:
        h (torch.Tensor): 物品的嵌入向量，形状为 [N, D]，其中 N 是物品数量，D 是嵌入维度
        noise_ratio (float): 加噪声的比例，值在 [0, 1] 之间

        返回:
        torch.Tensor: 加噪声后的嵌入向量
        """
        # 获取物品数量 N 和嵌入维度 D
        N, D = h.size()

        # 计算需要替换的物品数量
        num_items = int(N * noise_ratio)

        # 随机选择需要加噪的物品索引
        indices = torch.randint(0, N, (num_items,))

        # 随机选择需要替换的特征（注意：这里是随机选取物品，而不是每个特征维度）
        for idx in indices:
            random_item_idx = torch.randint(0, N, (1,))  # 随机选择其他物品
            h[idx] = h[random_item_idx]  # 用另一个物品的特征替换原特征
            h_f[idx] = h_f[random_item_idx]

        return h, h_f

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        # user_embeddings, item_embeddings, h_t, h_v, h_s, h_f, m_embeds, content_embeds = self.forward(IsTrain = True)
        user_embeddings, item_embeddings, h_v, h_t, h_f, m_embeds, content_embeds = self.forward(IsTrain=True)
        # h_t, h_v, h_f = self.add_noise_to_modalities(h_t, h_v, h_f, 0.80)
        self.build_item_graph = False

        # Multi content
        m_embeds_users, m_embeds_items = torch.split(m_embeds, [self.n_users, self.n_items], dim=0)
        # ID
        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
        cl_loss_mm = self.InfoNCE(m_embeds_items[pos_items], content_embeds_items[pos_items], 0.2) + self.InfoNCE(m_embeds_users[users], content_embeds_user[users], 0.2)

        #s_user_embeds = torch.sparse.mm(self.R, h_s)
        #cl_loss_s = self.InfoNCE(h_s[pos_items], content_embeds_items[pos_items], 0.2) + self.InfoNCE(s_user_embeds[users], content_embeds_user[users], 0.2)

        cl_loss = cl_loss_mm

        i_idx = torch.unique(torch.cat((pos_items, neg_items)), return_inverse=True, sorted=False)

        i_id = i_idx[0]

        # text
        label_prediction_t = self.label_prediction(h_t[i_id], h_t)
        # visual
        label_prediction_v = self.label_prediction(h_v[i_id], h_v)
        # session
        # label_prediction_s = self.label_prediction(h_s[i_id], h_s)

        label_prediction_f = self.label_prediction(h_f[i_id], h_f)

        # mm_postive_s, s_postive_s = self.generate_pesudo_labels(label_prediction_v, label_prediction_f, label_prediction_s)
        # neighbor_dis_loss_1 = self.neighbor_discrimination(mm_postive_s, s_postive_s, h_s[i_id], h_s)

        mm_postive_v, s_postive_v = self.generate_pesudo_labels(label_prediction_t, label_prediction_f, label_prediction_v)
        neighbor_dis_loss_2 = self.neighbor_discrimination(mm_postive_v, s_postive_v, h_v[i_id], h_v)

        mm_postive_t, s_postive_t = self.generate_pesudo_labels(label_prediction_v, label_prediction_f, label_prediction_t)
        neighbor_dis_loss_3 = self.neighbor_discrimination(mm_postive_t, s_postive_t, h_t[i_id], h_t)

        mm_postive_f, s_postive_f = self.generate_pesudo_labels(label_prediction_t, label_prediction_v, label_prediction_f)
        neighbor_dis_loss_4 = self.neighbor_discrimination(mm_postive_f, s_postive_f, h_f[i_id], h_f)
        '''
        
        mm_postive_s, s_postive_s = self.generate_pesudo_labels(label_prediction_t, label_prediction_v, label_prediction_f, label_prediction_s)
        neighbor_dis_loss_1 = self.neighbor_discrimination(mm_postive_s, s_postive_s, h_s[i_id], h_s)
        
        mm_postive_v, s_postive_v = self.generate_pesudo_labels(label_prediction_t, label_prediction_s, label_prediction_f, label_prediction_v)
        neighbor_dis_loss_2 = self.neighbor_discrimination(mm_postive_v, s_postive_v, h_v[i_id], h_v)
        
        mm_postive_t, s_postive_t = self.generate_pesudo_labels(label_prediction_v, label_prediction_s, label_prediction_f, label_prediction_t)
        neighbor_dis_loss_3 = self.neighbor_discrimination(mm_postive_t, s_postive_t, h_t[i_id], h_t)

        mm_postive_f, s_postive_f = self.generate_pesudo_labels(label_prediction_v, label_prediction_s, label_prediction_t, label_prediction_f)
        neighbor_dis_loss_4 = self.neighbor_discrimination(mm_postive_f, s_postive_f, h_f[i_id], h_f)
        '''

        neighbor_dis_loss = (neighbor_dis_loss_3 + neighbor_dis_loss_2 + neighbor_dis_loss_4) / 3.0

        weight_v, weight_t, weight_f = self.calculate_modal_weights(h_v, h_t, h_f)

        weight_t_density = torch.clamp(weight_t, 0.8, 1.5)
        weight_v_density = torch.clamp(weight_v, 0.8, 1.5)
        # weight_s_density = torch.clamp(weight_s, 0.8, 1.5)
        weight_f_density = torch.clamp(weight_f, 0.8, 1.5)

        ia_embeddings = item_embeddings + (weight_f_density * h_f + weight_t_density * h_t + weight_v_density * h_v)
        
        u_g_embeddings = user_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        return batch_mf_loss + self.neighbor_weight * (neighbor_dis_loss) + cl_loss * 0.001 # + KL_loss * self.kl_weight


    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embeddings, item_embeddings, h_v, h_t, h_f = self.forward(False) #

        user_e = user_embeddings[user, :]
        i_embedding = (h_v + h_t + h_f)/ 3.0
        all_item_e = item_embeddings + i_embedding 
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score


    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss