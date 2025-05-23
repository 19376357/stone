import torch
import numpy as np
from torchvision import models, transforms
import torch.nn as nn
from torch.nn.functional import cosine_similarity
from PIL import Image

import math

def seed_filling(image, diag=False):
    '''
    用Seed-Filling算法标记图片中的连通域.

    Parameters
    ----------
    image : ndarray, shape (nrow, ncol)
        图片数组,零值表示背景,非零值表示特征.

    diag : bool
        指定邻域是否包含四个对角.

    Returns
    -------
    labelled : ndarray, shape (nrow, ncol), dtype int
        表示连通域标签的数组,0表示背景,从1开始表示标签.

    nlabel : int
        连通域的个数.
    '''
    # 用-1表示未被标记过的特征像素.
    image = np.asarray(image, dtype=bool)
    nrow, ncol = image.shape
    labelled = np.where(image, -1, 0)

    # 指定邻域的范围.
    if diag:
        offsets = [
            (-1, -1), (-1, 0), (-1, 1),(0, -1),
            (0, 1), (1, -1), (1, 0), (1, 1)
        ]
    else:
        offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    def get_neighbor_indices(row, col):
        '''获取(row, col)位置邻域的下标.'''
        for (dx, dy) in offsets:
            x = row + dx
            y = col + dy
            if 0 <= x < nrow and 0 <= y < ncol:
                yield x, y

    label = 1
    for row in range(nrow):
        for col in range(ncol):
            # 跳过背景像素和已经标记过的特征像素.
            if labelled[row, col] != -1:
                continue
            # 标记当前位置和邻域内的特征像素.
            current_indices = []
            labelled[row, col] = label
            for neighbor_index in get_neighbor_indices(row, col):
                if labelled[neighbor_index] == -1:
                    labelled[neighbor_index] = label
                    current_indices.append(neighbor_index)
            # 不断寻找与特征像素相邻的特征像素并加以标记,直至再找不到特征像素.
            while current_indices:
                current_index = current_indices.pop()
                labelled[current_index] = label
                for neighbor_index in get_neighbor_indices(*current_index):
                    if labelled[neighbor_index] == -1:
                        labelled[neighbor_index] = label
                        current_indices.append(neighbor_index)
            label += 1

    return labelled, label - 1

def frequency(lst):
    freq = {}
    for item in lst:
        if item in freq:
            freq[item] += 1
        else:
            freq[item] = 1
    return freq
def entropy(lst):
    freq = frequency(lst)
    ent = 0
    for key, value in freq.items():
        prob = value / len(lst)
        ent += -prob * math.log2(prob)
    return ent

class TopologicallyRegularizedLoss:

    def __init__(self, lam=1., toposig_kwargs=None):

        self.lam = lam
        self.topo_sig = TopologicalSignatureDistance(match_edges='symmetric')

        self.resnet = models.resnet18(pretrained=True)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
          ])
        self.resnet.eval()

    @staticmethod
    def _compute_distance_matrix(x, p=2):
        x_flat = x.view(x.size(0), -1)
        distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=p)
        return distances

    def topo_loss(self, x, latent):
        if isinstance(x, np.ndarray):
            x = Image.fromarray(x)
        elif isinstance(x, Image.Image):
            pass
        else:
            raise TypeError("x must be either np.array or PIL.image")
        if isinstance(latent, np.ndarray):
            latent = Image.fromarray(latent)
        elif isinstance(latent, Image.Image):
            pass
        else:
            raise TypeError("latent must be either np.array or PIL.image")
        x_transformed = self.transform(x).unsqueeze(0)
        latent_transformed = self.transform(latent).unsqueeze(0)
        latent_transformed = self.resnet(latent_transformed)

        x_distances = self._compute_distance_matrix(x_transformed)
        latent_distances = self._compute_distance_matrix(latent_transformed)

        dimensions = x_transformed.size()
        # If we have an image dataset, normalize using theoretical maximum
        batch_size, ch, b, w = dimensions
        # Compute the maximum distance we could get in the data space (this
        # is only valid for images wich are normalized between -1 and 1)
        max_distance = (2**2 * ch * b * w) ** 0.5
        x_distances = x_distances / max_distance
        latent_distances = latent_distances / max_distance

        topo_error, topo_error_components = self.topo_sig(x_distances, latent_distances)

    def reconstruct_loss(self, x, latent):
        if isinstance(x, np.ndarray):
            x = Image.fromarray(x)
        elif isinstance(x, Image.Image):
            pass
        else:
            raise TypeError("x must be either np.array or PIL.image")
        if isinstance(latent, np.ndarray):
            latent = Image.fromarray(latent)
        elif isinstance(latent, Image.Image):
            pass
        else:
            raise TypeError("latent must be either np.array or PIL.image")

        x_transformed = self.transform(x).unsqueeze(0)
        latent_transformed = self.transform(latent).unsqueeze(0)

        cos_sim = cosine_similarity(self.resnet(x_transformed), self.resnet(latent_transformed), dim=1)
        return cos_sim.detach().numpy()[0]

class TopologicalSignatureDistance(nn.Module):
    """Topological signature."""

    def __init__(self, sort_selected=False, use_cycles=False,
                 match_edges=None):
        """Topological signature computation.

        Args:
            p: Order of norm used for distance computation
            use_cycles: Flag to indicate whether cycles should be used
                or not.
        """
        super().__init__()
        self.use_cycles = use_cycles

        self.match_edges = match_edges

        # if use_cycles:
        #     use_aleph = True
        # else:
        #     if not sort_selected and match_edges is None:
        #         use_aleph = True
        #     else:
        #         use_aleph = False

        # if use_aleph:
        #     print('Using aleph to compute signatures')
        ##self.signature_calculator = AlephPersistenHomologyCalculation(
        ##    compute_cycles=use_cycles, sort_selected=sort_selected)
        # else:
        print('Using python to compute signatures')
        self.signature_calculator = PersistentHomologyCalculation()

    def _get_pairings(self, distances):
        pairs_0, pairs_1 = self.signature_calculator(
            distances.detach().cpu().numpy())

        return pairs_0, pairs_1

    def _select_distances_from_pairs(self, distance_matrix, pairs):
        # Split 0th order and 1st order features (edges and cycles)
        pairs_0, pairs_1 = pairs
        selected_distances = distance_matrix[(pairs_0[:, 0], pairs_0[:, 1])]

        if self.use_cycles:
            edges_1 = distance_matrix[(pairs_1[:, 0], pairs_1[:, 1])]
            edges_2 = distance_matrix[(pairs_1[:, 2], pairs_1[:, 3])]
            edge_differences = edges_2 - edges_1

            selected_distances = torch.cat(
                (selected_distances, edge_differences))

        return selected_distances

    @staticmethod
    def sig_error(signature1, signature2):
        """Compute distance between two topological signatures."""
        return ((signature1 - signature2) ** 2).sum(dim=-1)

    @staticmethod
    def _count_matching_pairs(pairs1, pairs2):
        def to_set(array):
            return set(tuple(elements) for elements in array)

        return float(len(to_set(pairs1).intersection(to_set(pairs2))))

    @staticmethod
    def _get_nonzero_cycles(pairs):
        all_indices_equal = np.sum(pairs[:, [0]] == pairs[:, 1:], axis=-1) == 3
        return np.sum(np.logical_not(all_indices_equal))

    # pylint: disable=W0221
    def forward(self, distances1, distances2):
        """Return topological distance of two pairwise distance matrices.

        Args:
            distances1: Distance matrix in space 1
            distances2: Distance matrix in space 2

        Returns:
            distance, dict(additional outputs)
        """
        pairs1 = self._get_pairings(distances1)
        pairs2 = self._get_pairings(distances2)

        distance_components = {
            'metrics.matched_pairs_0D': self._count_matching_pairs(
                pairs1[0], pairs2[0])
        }
        # Also count matched cycles if present
        if self.use_cycles:
            distance_components['metrics.matched_pairs_1D'] = \
                self._count_matching_pairs(pairs1[1], pairs2[1])
            nonzero_cycles_1 = self._get_nonzero_cycles(pairs1[1])
            nonzero_cycles_2 = self._get_nonzero_cycles(pairs2[1])
            distance_components['metrics.non_zero_cycles_1'] = nonzero_cycles_1
            distance_components['metrics.non_zero_cycles_2'] = nonzero_cycles_2

        if self.match_edges is None:
            sig1 = self._select_distances_from_pairs(distances1, pairs1)
            sig2 = self._select_distances_from_pairs(distances2, pairs2)
            distance = self.sig_error(sig1, sig2)

        elif self.match_edges == 'symmetric':
            print(distances1)
            print(pairs1)
            sig1 = self._select_distances_from_pairs(distances1, pairs1)
            sig2 = self._select_distances_from_pairs(distances2, pairs2)
            print(distances2)
            print(pairs2)
            # Selected pairs of 1 on distances of 2 and vice versa
            sig1_2 = self._select_distances_from_pairs(distances2, pairs1)
            sig2_1 = self._select_distances_from_pairs(distances1, pairs2)

            distance1_2 = self.sig_error(sig1, sig1_2)
            distance2_1 = self.sig_error(sig2, sig2_1)

            distance_components['metrics.distance1-2'] = distance1_2
            distance_components['metrics.distance2-1'] = distance2_1

            distance = distance1_2 + distance2_1

        elif self.match_edges == 'random':
            # Create random selection in oder to verify if what we are seeing
            # is the topological constraint or an implicit latent space prior
            # for compactness
            n_instances = len(pairs1[0])
            pairs1 = torch.cat([
                torch.randperm(n_instances)[:, None],
                torch.randperm(n_instances)[:, None]
            ], dim=1)
            pairs2 = torch.cat([
                torch.randperm(n_instances)[:, None],
                torch.randperm(n_instances)[:, None]
            ], dim=1)

            sig1_1 = self._select_distances_from_pairs(
                distances1, (pairs1, None))
            sig1_2 = self._select_distances_from_pairs(
                distances2, (pairs1, None))

            sig2_2 = self._select_distances_from_pairs(
                distances2, (pairs2, None))
            sig2_1 = self._select_distances_from_pairs(
                distances1, (pairs2, None))

            distance1_2 = self.sig_error(sig1_1, sig1_2)
            distance2_1 = self.sig_error(sig2_1, sig2_2)
            distance_components['metrics.distance1-2'] = distance1_2
            distance_components['metrics.distance2-1'] = distance2_1

            distance = distance1_2 + distance2_1

        return distance, distance_components

'''
Methods for calculating lower-dimensional persistent homology.
'''

class UnionFind:
    '''
    An implementation of a Union--Find class. The class performs path
    compression by default. It uses integers for storing one disjoint
    set, assuming that vertices are zero-indexed.
    '''

    def __init__(self, n_vertices):
        '''
        Initializes an empty Union--Find data structure for a given
        number of vertices.
        '''

        self._parent = np.arange(n_vertices, dtype=int)

    def find(self, u):
        '''
        Finds and returns the parent of u with respect to the hierarchy.
        '''

        if self._parent[u] == u:
            return u
        else:
            # Perform path collapse operation
            self._parent[u] = self.find(self._parent[u])
            return self._parent[u]

    def merge(self, u, v):
        '''
        Merges vertex u into the component of vertex v. Note the
        asymmetry of this operation.
        '''

        if u != v:
            self._parent[self.find(u)] = self.find(v)

    def roots(self):
        '''
        Generator expression for returning roots, i.e. components that
        are their own parents.
        '''

        for vertex, parent in enumerate(self._parent):
            if vertex == parent:
                yield vertex

class PersistentHomologyCalculation:
    def __call__(self, matrix):

        n_vertices = matrix.shape[0]
        uf = UnionFind(n_vertices)

        triu_indices = np.triu_indices_from(matrix)
        edge_weights = matrix[triu_indices]
        edge_indices = np.argsort(edge_weights, kind='stable')

        # 1st dimension: 'source' vertex index of edge
        # 2nd dimension: 'target' vertex index of edge
        persistence_pairs = []

        for edge_index, edge_weight in \
                zip(edge_indices, edge_weights[edge_indices]):

            u = triu_indices[0][edge_index]
            v = triu_indices[1][edge_index]

            younger_component = uf.find(u)
            older_component = uf.find(v)

            # Not an edge of the MST, so skip it
            if younger_component == older_component:
                continue
            elif younger_component > older_component:
                uf.merge(v, u)
            else:
                uf.merge(u, v)

            if u < v:
                persistence_pairs.append((u, v))
            else:
                persistence_pairs.append((v, u))

        # Return empty cycles component
        return np.array(persistence_pairs), np.array([])

