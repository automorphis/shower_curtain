import numpy as np

class R2_Graph:

    def __init__(self, adj_matrix, coords):

        self.adj_mat = adj_matrix
        self.coords = coords

        self.n = len(self.adj_mat.shape)
        self.degs = np.sum(self.adj_mat, axis=0)
        self.e = np.sum(self.degs)//2

        self.higher_degs = [
            np.ones(self.n, dtype=int),
            np.zeros(self.n, dtype=int),
            self.degs
        ]
        self._higher_degs_stack = None

        self.adj_mat_pows = [np.eye(self.n,dtype=int), self.adj_mat]
        self._adj_mat_pows_stack = None

        self.adj_mat_pow_diags = [np.ones(self.n, dtype=int), np.zeros(self.n, dtype=int)]
        self._adj_mat_pow_diags_stack = None

        self.cycles = [
            np.zeros((0,0),dtype=int),
            np.zeros((1,0),dtype=int)
        ]




    def calc_adj_matrix_power_up_to(self, k):

        if k < 0:
            raise ValueError

        if len(self.adj_mat_pows) < k+1:

            curr = self.adj_mat_pows[-1]
            for _ in range(len(self.adj_mat_pows), k+1):
                curr = np.matmul(self.adj_mat, curr)
                self.adj_mat_pows.append(curr)
                self.adj_mat_pow_diags.append(np.diagonal(curr))

        return self.adj_mat_pows[:k+1]


    def _stack_adj_mat_pows(self, highest_pow):

        self.calc_adj_matrix_power_up_to(highest_pow)

        if self._adj_mat_pows_stack is not None and self._adj_mat_pows_stack.shape[2] >= highest_pow:
            return self._adj_mat_pows_stack[:,:,:highest_pow+1], self._adj_mat_pow_diags_stack[:,:highest_pow+1]

        self._adj_mat_pows_stack = np.stack(self.adj_mat_pows[ : highest_pow + 1], axis=2)
        self._adj_mat_pow_diags_stack = np.stack(self.adj_mat_pow_diags, axis=1)

        return self._adj_mat_pows_stack, self._adj_mat_pow_diags_stack


    def is_connected(self):

        stack, _ = self._stack_adj_mat_pows(self.n)

        return np.all(np.sum(stack, axis=2) > 0)


    def calc_cycles_up_to(self, length):

        _, diag_stack = self._stack_adj_mat_pows(length)

        self._higher_degs_stack = np.concatenate([
            np.stack(self.higher_degs, axis=1),
            np.zeros((self.n, length - len(self.higher_degs) + 1), dtype=int)
        ], axis=1)

        degs_stack = self._higher_degs_stack

        for l in range(3, length+1):

            num_loops = np.sum(degs_stack[:, 1:l] * diag_stack[:, l-1:0:-1], axis=0)

            degs = diag_stack[:,l] - num_loops
            self.higher_degs.append(degs)
            self._higher_degs_stack[:,l] = degs

            cycles = np.zeros((l, np.sum(degs) // l), dtype=int)


        prods = stack[:,1:-1] * stack[:,1:-1:-1]

        num_loops = np.sum(prods, axis=1)

        num_cycles = stack[:,-1] - num_loops

        cycles = np.zeros((length, np.sum(num_cycles)//length), dtype=int)


