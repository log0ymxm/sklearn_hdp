from __future__ import division
from sklearn.base import BaseEstimator, TransformerMixin, ClusterMixin
import numpy as np
from scipy.special import gammaln
from sklearn.utils.validation import NotFittedError, check_is_fitted

class DefaultDict(dict):
    def __init__(self, v):
        self.v = v
        dict.__init__(self)

    def __getitem__(self, k):
        return dict.__getitem__(self, k) if k in self else self.v

    def update(self, d):
        dict.update(self, d)
        return self

class HierarchicalDirichletProcess(BaseEstimator, TransformerMixin, ClusterMixin):
    """
    Implements the Hierarchical Dirichlet Process.
    """
    def __init__(self, alpha=1.0, gamma=1.0, beta=0.5, max_iter=10, random_state=None, verbose=False):
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

        self.topics = None

        self.V = None # TODO rename self.vocab_size
        self.M = None # TODO rename self.document_size
        self.using_t = None # TODO rename self.table_indices
        self.using_k = None # TODO rename self.dish_indices
        self.n_kv = None

    def _setup(self, X):
        self.V = np.unique(np.array(X).flatten()).shape[0]
        self.M = len(X)

        # t: table index for document j
        #    t=0 means to draw from a new table
        # TODO rename self.table_indices
        self.using_t = [[0] for j in xrange(self.M)]

        # k: dish(topic) index
        #    k=0 means to draw a new dish
        self.using_k = [0]

        self.x_ji = X # vocabulary for each document and term
        self.k_jt = [np.zeros(1, dtype=int) for j in xrange(self.M)] # topics of document and table
        self.n_jt = [np.zeros(1, dtype=int) for j in xrange(self.M)] # number of terms for each table of document
        self.n_jtv = [[None] for j in xrange(self.M)]

        self.m = 0
        self.m_k = np.ones(1, dtype=int) # number of tables for each topic
        self.n_k = np.array([self.beta * self.V]) # number of terms for each topic ( + beta * V )
        self.n_kv = [DefaultDict(0)] # number of terms for each topic and vocabulary ( + beta )

        # table for each document and term (-1 means not-assigned)
        self.t_ji = [np.zeros(len(x_i), dtype=int) - 1 for x_i in X]

    def fit(self, X, y=None):
        self._setup(X)

        if self.verbose:
            print("| Iteration | Num. Topics | Perplexity |")
            print("|-----------+-------------+------------|")

        for i in range(self.max_iter):
            self._inference()
            if self.verbose:
                print("| %d | %d | %f |" % (i+1, len(self.topics), self.perplexity()))


    def partial_fit(self, X):
        # TODO
        pass

    def fit_predict(self, X):
        # predicts the majority topic for each doc
        # TODO
        pass

    def fit_transform(self, X):
        # TODO
        pass

    def transform(self, X):
        # TODO
        pass

    def get_params(self):
        # TODO
        pass

    def set_params(self):
        # TODO
        pass

    def predict(self, X):
        # TODO
        pass

    def predict_proba(self, X):
        # TODO
        # gives prob dist for each topic on the documents
        pass

    def perplexity(self, X=None):
        # TODO check perplexity with new docs??
        self._check_fit()

        phi = [DefaultDict(1.0 / self.V)] + self.worddist()
        theta = self.docdist()
        log_likelihood = 0
        N = 0
        for x_ji, p_jk in zip(self.x_ji, theta):
            for v in x_ji:
                word_prob = sum(p * p_kv[v] for p, p_kv in zip(p_jk, phi))
                log_likelihood -= np.log(word_prob)
            N += len(x_ji)
        return np.exp(log_likelihood / N)

    def _check_fit(self):
        is_fitted = hasattr(self, 'n_kv') # TODO

        # Called by partial_fit, before fitting.
        has_partial_fit = hasattr(self, 'partial_fit_')

        # Should raise an error if one does not fit before predicting
        if not (is_fitted or has_partial_fit):
            raise NotFittedError("Fit training data before predicting") # TODO

        # XXX
        # if is_fitted and X.shape[1] != self.M.shape[1]:
        #     raise ValueError("Training data and predicted data do "
        #                      "not have same number of features.")

    def _inference(self):
        for j, x_i in enumerate(self.x_ji):
            for i in xrange(len(x_i)):
                self._sampling_t(j, i)
        for j in xrange(self.M):
            for t in self.using_t[j]:
                if t != 0:
                     self._sampling_k(j, t)

    def worddist(self):
        """Return the topic-word distribution without new topic"""
        # TODO clean this up maybe...
        return [DefaultDict(self.beta / self.n_k[k]).update(
            (v, n_kv / self.n_k[k]) for v, n_kv in self.n_kv[k].iteritems())
                for k in self.using_k if k != 0]

    def docdist(self):
        """Return the document-topic distribution with new topic"""

        # am_k = effect from table-dish assignment
        am_k = np.array(self.m_k, dtype=float)
        am_k[0] = self.gamma
        am_k *= self.alpha / am_k[self.using_k].sum()

        theta = []
        for j, n_jt in enumerate(self.n_jt):
            p_jk = am_k.copy()
            for t in self.using_t[j]:
                if t==0:
                    continue
                k = self.k_jt[j][t]
                p_jk[k] += n_jt[t]
            p_jk = p_jk[self.using_k]
            theta.append(p_jk / p_jk.sum())

        return np.array(theta)

    def _dump(self, disp_x=False):
        if disp_x:
            print("x_ji:", self.x_ji)
        print("using_t:", self.using_t)
        print("t_ji:", self.t_ji)
        print("using_k:", self.using_k)
        print("k_jt:", self.k_jt)
        print("----")
        print("n_jt:", self.n_jt)
        print("n_jtv:", self.n_jtv)
        print("n_k:", self.n_k)
        print("n_kv:", self.n_kv)
        print("m:", self.m)
        print("m_k:", self.m_k)
        print()

    def _sampling_t(self, j, i):
        """sampling t (table) from posterior"""
        self._leave_from_table(j, i)

        v = self.x_ji[j][i]
        f_k = self._calc_f_k(v)
        assert f_k[0] == 0 # f_k[0] is a dummy and will be erased

        # sampling from posterior p(t_ji=t)
        p_t = self._calc_table_posterior(j, f_k)
        if len(p_t) > 1 and p_t[1] < 0:
            self._dump()
        t_new = self.using_t[j][np.random.multinomial(1, p_t).argmax()]
        if t_new == 0:
            p_k = self._calc_dish_posterior_w(f_k)
            k_new = self.using_k[np.random.multinomial(1, p_k).argmax()]
            if k_new == 0:
                k_new = self._add_new_dish()
            t_new = self._add_new_table(j, k_new)

        # increase counters
        self._seat_at_table(j, i, t_new)

    def _leave_from_table(self, j, i):
        t = self.t_ji[j][i]
        if t > 0:
            k = self.k_jt[j][t]
            assert k > 0

            # decrease counters
            v = self.x_ji[j][i]
            self.n_kv[k][v] -= 1
            self.n_k[k] -= 1
            self.n_jt[j][t] -= 1
            self.n_jtv[j][t][v] -= 1

            if self.n_jt[j][t] == 0:
                self._remove_table(j, t)

    def _remove_table(self, j, t):
        """Remove the table where all guests are gone"""
        k = self.k_jt[j][t]
        self.using_t[j].remove(t)
        self.m_k[k] -= 1
        self.m -= 1
        assert self.m_k[k] >= 0
        if self.m_k[k] == 0:
            # remove topic (dish) where all tables are gone
            self.using_k.remove(k)

    def _calc_f_k(self, v):
        return [n_kv[v] for n_kv in self.n_kv] / self.n_k

    def _calc_table_posterior(self, j, f_k):
        using_t = self.using_t[j]
        p_t = self.n_jt[j][using_t] * f_k[self.k_jt[j][using_t]]
        p_x_ji = np.inner(self.m_k, f_k) + self.gamma / self.V
        p_t[0] = p_x_ji * self.alpha / (self.gamma + self.m)
        # print "un-normalized p_t = ", p_t
        return p_t / p_t.sum()

    def _seat_at_table(self, j, i, t_new):
        assert t_new in self.using_t[j]
        self.t_ji[j][i] = t_new
        self.n_jt[j][t_new] += 1

        k_new = self.k_jt[j][t_new]
        self.n_k[k_new] += 1

        v = self.x_ji[j][i]
        self.n_kv[k_new][v] += 1
        self.n_jtv[j][t_new][v] += 1

    def _add_new_table(self, j, k_new):
        """Assign guest x_ji to a new table and draw topic (dish) of the table"""
        assert k_new in self.using_k
        for t_new, t in enumerate(self.using_t[j]):
            if t_new != t:
                break
        else:
            t_new = len(self.using_t[j])
            self.n_jt[j].resize(t_new+1)
            self.k_jt[j].resize(t_new+1)
            self.n_jtv[j].append(None)

        self.using_t[j].insert(t_new, t_new)
        self.n_jt[j][t_new] = 0 # to make sure
        self.n_jtv[j][t_new] = DefaultDict(0)

        self.k_jt[j][t_new] = k_new
        self.m_k[k_new] += 1
        self.m += 1

        return t_new

    def _calc_dish_posterior_w(self, f_k):
        """calculate dish (topic) posterior when one word is removed"""
        p_k = (self.m_k * f_k)[self.using_k]
        p_k[0] = self.gamma / self.V
        return p_k / p_k.sum()

    def _sampling_k(self, j, t):
        """sampling k (dish=topic) from posterior"""
        self._leave_from_dish(j, t)

        # sampling of k
        p_k = self._calc_dish_posterior_t(j, t)
        k_new = self.using_k[np.random.multinomial(1, p_k).argmax()]
        if k_new == 0:
            k_new = self._add_new_dish()

        self._seat_at_dish(j, t, k_new)

    def _leave_from_dish(self, j, t):
        """
        This makes the table leave from its dish and only the table counter decrease.
        The word counters (n_k and n_kv) stay.
        """
        k = self.k_jt[j][t]
        assert k > 0
        assert self.m_k[k] > 0
        self.m_k[k] -= 1
        self.m -= 1
        if self.m_k[k] == 0:
            self.using_k.remove(k)
            self.k_jt[j][t] = 0

    def _calc_dish_posterior_t(self, j, t):
        """calculate dish(topic) posterior when one table is removed"""
        k_old = self.k_jt[j][t] # it may be zero (means a removed dish)
        #print("V=", self.V, "beta=", self.beta, "n_k=", self.n_k)
        Vbeta = self.V * self.beta
        n_k = self.n_k.copy()
        n_jt = self.n_jt[j][t]
        n_k[k_old] -= n_jt
        n_k = n_k[self.using_k]
        log_p_k = np.log(self.m_k[self.using_k]) + gammaln(n_k) - gammaln(n_k + n_jt)
        log_p_k_new = np.log(self.gamma) + gammaln(Vbeta) - gammaln(Vbeta + n_jt)
        #print("log_p_k_new+=gammaln(", Vbeta,") - gammaln(", Vbeta + n_jt,")")

        gammaln_beta = gammaln(self.beta)
        for w, n_jtw in self.n_jtv[j][t].iteritems():
            assert n_jtw >= 0
            if n_jtw == 0:
                continue
            n_kw = np.array([n.get(w, self.beta) for n in self.n_kv])
            n_kw[k_old] -= n_jtw
            n_kw = n_kw[self.using_k]
            n_kw[0] = 1 # dummy for logarithm's warning
            if np.any(n_kw <= 0):
                print(n_kw) # for debug
            log_p_k += gammaln(n_kw + n_jtw) - gammaln(n_kw)
            log_p_k_new += gammaln(self.beta + n_jtw) - gammaln_beta
            #print("log_p_k_new+=gammaln(",self.beta + n_jtw,") - gammaln(",self.beta,"), w=", w)
        log_p_k[0] = log_p_k_new
        #print("un-normalized p_k = ", np.exp(log_p_k))
        p_k = np.exp(log_p_k - log_p_k.max())
        return p_k / p_k.sum()

    def _seat_at_dish(self, j, t, k_new):
        self.m += 1
        self.m_k[k_new] += 1

        k_old = self.k_jt[j][t] # it may be zero (means a removed dish)
        if k_new != k_old:
            self.k_jt[j][t] = k_new

            n_jt = self.n_jt[j][t]
            if k_old != 0:
                self.n_k[k_old] -= n_jt
            self.n_k[k_new] += n_jt
            for v, n in self.n_jtv[j][t].iteritems():
                if k_old != 0:
                    self.n_kv[k_old][v] -= n
                self.n_kv[k_new][v] += n

    def _add_new_dish(self):
        """This is commonly used by sampling_t and sampling_k."""
        for k_new, k in enumerate(self.using_k):
            if k_new != k:
                break
        else:
            k_new = len(self.using_k)
            if k_new >= len(self.n_kv):
                self.n_k = np.resize(self.n_k, k_new + 1)
                self.m_k = np.resize(self.m_k, k_new + 1)
                self.n_kv.append(None)
            assert k_new == self.using_k[-1] + 1
            assert k_new < len(self.n_kv)

        self.using_k.insert(k_new, k_new)
        self.n_k[k_new] = self.beta * self.V
        self.m_k[k_new] = 0
        self.n_kv[k_new] = DefaultDict(self.beta)
        return k_new

    def _output_summary(self, voca, fp=None):
        # TODO
        if fp==None:
            import sys
            fp = sys.stdout
        K = len(self.using_k) - 1
        kmap = dict((k,i-1) for i, k in enumerate(self.using_k))
        dishcount = numpy.zeros(K, dtype=int)
        wordcount = [DefaultDict(0) for k in xrange(K)]
        for j, x_ji in enumerate(self.x_ji):
            for v, t in zip(x_ji, self.t_ji[j]):
                k = kmap[self.k_jt[j][t]]
                dishcount[k] += 1
                wordcount[k][v] += 1

        phi = self.worddist()
        for k, phi_k in enumerate(phi):
            fp.write("\n-- topic: %d (%d words)\n" % (self.using_k[k+1], dishcount[k]))
            for w in sorted(phi_k, key=lambda w:-phi_k[w])[:20]:
                fp.write("%s: %f (%d)\n" % (voca[w], phi_k[w], wordcount[k][w]))

        fp.write("--- document-topic distribution\n")
        theta = self.docdist()
        for j, theta_j in enumerate(theta):
            fp.write("%d\t%s\n" % (j, "\t".join("%.3f" % p for p in theta_j[1:])))

        fp.write("--- dishes for document\n")
        for j, using_t in enumerate(self.using_t):
            fp.write("%d\t%s\n" % (j, "\t".join(str(self.k_jt[j][t]) for t in using_t if t>0)))
