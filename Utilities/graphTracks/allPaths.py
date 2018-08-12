import scipy.sparse
import scipy

import Utilities.general.ismember


__author__ = 'edwardharry'


def allPaths(wt=scipy.array([[1]]), startnode=scipy.array([[0]]), endnode=scipy.array([0]), n=None, startN=None,
             revisitNodes=None):
    N = wt.shape
    assert (N[0] == N[1])
    N = N[0]
    wt = scipy.sparse.lil_matrix(wt > 0)
    lastpath = scipy.copy(startnode)
    paths = list()

    if n is None:
        n = N

    if startN is None:
        startN = 1

    if revisitNodes is None:
        revisitNodes = False

    nDiag = scipy.sparse.spdiags(scipy.arange(1, N + 1), 0, N, N)

    for i in range(startN, n):
        sizeLastPath = lastpath.shape
        nextmove = wt[lastpath[:, i - 1], :]

        d = scipy.sparse.spdiags(scipy.arange(1, sizeLastPath[0] + 1), 0, sizeLastPath[0], sizeLastPath[0])
        if not revisitNodes:
            nrows = d * scipy.ones(sizeLastPath)
            nrows = nrows.astype(scipy.int64).__add__(-1)
            nextmove[(nrows.flatten(), lastpath.flatten())] = False

        if not any(nextmove.toarray().flatten()):
            break

        nextmoverow = d * nextmove
        nextmovecol = nextmove * nDiag
        nextmoverow = nextmoverow.toarray().transpose().flatten()
        nextmovecol = nextmovecol.toarray().transpose().flatten()

        nextmoverow = nextmoverow[~(nextmoverow == 0)]
        nextmovecol = nextmovecol[~(nextmovecol == 0)]

        nextmoverow = nextmoverow.__add__(-1)
        nextmovecol = nextmovecol.__add__(-1)

        lastpath = scipy.concatenate((lastpath[nextmoverow, :], scipy.array([nextmovecol]).transpose()), 1)

        reachedend = Utilities.general.ismember.ismember(lastpath[:, i], endnode)
        reachedend = scipy.array(reachedend).astype(scipy.bool_)
        if any(reachedend):
            paths.append(lastpath[reachedend, :])

    return paths






