import numpy as np
from PIL import Image

DIFF_EPSILON_SQ = 1e-6
DIFF_ITERS = 20
DIFF_TAU = 0.5

def IsotropicDiffMatrixMultiChannel(inputMatrix):
  resultX = np.zeros(inputMatrix.shape[:2])
  resultY = np.zeros(inputMatrix.shape[:2])

  dY, dX = np.gradient(inputMatrix, axis=(0, 1))
  
  gradY = inputMatrix[1:, ...] - inputMatrix[:(inputMatrix.shape[0] - 1), ...]
  gradX = 0.5 * (dX[1:, ...] + dX[:(inputMatrix.shape[0] - 1), ...])
  resultY[:(inputMatrix.shape[0] - 1), ...] = ((gradX * gradX) + (gradY * gradY)).sum(axis=2)

  resultY = 1.0 / np.sqrt(resultY + DIFF_EPSILON_SQ)

  gradY = 0.5 * (dY[:, 1:, :] + dY[:, :(inputMatrix.shape[1] - 1), :])
  gradX = inputMatrix[:, 1:, :] - inputMatrix[:, :(inputMatrix.shape[1] - 1), :]
  resultX[:, :(inputMatrix.shape[1] - 1)] = ((gradX * gradX) + (gradY * gradY)).sum(axis=2)

  resultX = 1.0 / np.sqrt(resultX + DIFF_EPSILON_SQ)

  return resultX, resultY

def IsotropicDiffuseSemiImplicit(inputMatrix, gX, gY):
  result = np.zeros(inputMatrix.shape[:2])
  M = np.zeros(inputMatrix.shape[:2])
  Y = np.zeros(inputMatrix.shape[:2])
  I = np.zeros(inputMatrix.shape[:2])
  J = np.zeros(inputMatrix.shape[:2])
  #K = np.zeros(inputMatrix.shape[:2])

  temp = (gX[:, :(gX.shape[1] - 1)] + gX[:, 1:]) * DIFF_TAU
  J[:, :(gX.shape[1] - 1)] = -temp
  #K[:, 1:] = -temp
  I[:, 0] = 1 + temp[:, 0]
  I[:, 1:(gX.shape[1] - 1)] = 1 + temp[:, 1:] + temp[:, :(temp.shape[1] - 1)]
  I[:, -1] = 1 + temp[:, -1]

  M[0, 0] = 1 / I[0, 0]
  Y[0, 0] = inputMatrix[0, 0]
  i = 0
  size_loop = (result.shape[0] * result.shape[1]) - 1
  
  while True:
    coords = np.unravel_index(i, result.shape)
    #l = K[coords] * M[coords]
    l = J[coords] * M[coords]
    i += 1
    coords_plusone = np.unravel_index(i, result.shape)
    M[coords_plusone] = 1 / (I[coords_plusone] - (l * J[coords]))
    Y[coords_plusone] = inputMatrix[coords_plusone] - (l * Y[coords])
    if i == size_loop:
      break

  coords = np.unravel_index(size_loop, result.shape)
  v = Y[coords] * M[coords]
  result[coords] = v
  for j in range(size_loop - 1, -1, -1):
    coords = np.unravel_index(j, result.shape)
    v = (Y[coords] - (v * J[coords])) * M[coords]
    result[coords] = v

  I = np.zeros(inputMatrix.shape[:2])
  J = np.zeros(inputMatrix.shape[:2])
  #K = np.zeros(inputMatrix.shape[:2])

  temp = (gY[:(gY.shape[0] - 1), :] + gY[1:, :]) * DIFF_TAU
  J[:(gY.shape[0] - 1), :] = -temp
  #J.flat[:(temp.shape[0] * temp.shape[1])] = (-temp).T.flat
  #K[1:, :] = -temp
  I[0, :] = 1 + temp[0, :]
  #I.flat[:temp.shape[1]] = (1 + temp[0, :]).T.flat
  I[1:(gY.shape[0] - 1), :] = 1 + temp[1:, :] + temp[:(temp.shape[0] - 1), :]
  #I.flat[temp.shape[1]:(((temp.shape[0] - 1) * temp.shape[1]) + temp.shape[1])] = (1 + temp[1:, :] + temp[:(temp.shape[0] - 1), :]).T.flat
  I[-1, :] = 1 + temp[-1, :]
  #I.flat[(((temp.shape[0] - 1) * temp.shape[1]) + temp.shape[1]):] = (1 + temp[-1, :]).T.flat

  I = I.T.reshape(I.shape)
  J = J.T.reshape(J.shape)

  M[0, 0] = 1 / I[0, 0]
  Y[0, 0] = inputMatrix[0, 0]
  i = 0
  y = 0
  x = 0

  while True:
    coords = np.unravel_index(i, result.shape)
    #l = K[coords] * M[coords]
    l = J[coords] * M[coords]
    i += 1
    y += 1
    if y == result.shape[0]:
      y = 0
      x += 1
    coords_plusone = np.unravel_index(i, result.shape)
    M[coords_plusone] = 1 / (I[coords_plusone] - (l * J[coords]))
    Y[coords_plusone] = inputMatrix[y, x] - (l * Y[coords])
    if i == size_loop:
      break

  coords = np.unravel_index(size_loop, result.shape)
  v = Y[coords] * M[coords]
  result[coords] += v
  x = result.shape[1] - 1
  y = result.shape[0] - 1
  for j in range(size_loop - 1, -1, -1):
    y -= 1
    if y < 0:
      y = result.shape[0] - 1
      x -= 1
    coords = np.unravel_index(j, result.shape)
    v = (Y[coords] - (v * J[coords])) * M[coords]
    result[y, x] += v

  return 0.5 * result 


def LocalFlowVar(flow):
  flowShape = flow.shape
  flowShape0 = tuple(flowShape[:2])

  flowShape = ((flowShape[0] >> 1) + 1, (flowShape[1] >> 1) + 1, flowShape[2])
  
  flowDS = np.zeros(flowShape)
  stats = np.zeros((flowShape[0], flowShape[1], 2 * flowShape[2]))
  for z in range(flowShape[2]):
    im = Image.fromarray(flow[..., z])
    im = im.resize(flowShape[:2], Image.ANTIALIAS)
    flowDS[..., z] = np.array(im)

    stats[..., z] = np.array(im)
    im = Image.fromarray(flow[..., z] * flow[..., z])
    im = im.resize(flowShape[:2], Image.ANTIALIAS)
    stats[..., z + flowShape[2]] = np.array(im)

  gX, gY = IsotropicDiffMatrixMultiChannel(flowDS)

  for k in range(DIFF_ITERS):
    for z in range(stats.shape[-1]):
      stats[..., z] = IsotropicDiffuseSemiImplicit(stats[..., z], gX, gY)

  flowVar = stats[..., 2:].sum(axis=2) - (stats[..., :2] * stats[..., :2]).sum(axis=2)
  flowVar[flowVar < 0] = 0
  flowVar = np.sqrt(flowVar)
  im = Image.fromarray(flowVar)
  im = im.resize(flowShape0, Image.ANTIALIAS)

  return np.array(im)

if __name__ == '__main__':
  import h5py
  with h5py.File('test_flow_data.h5', 'r') as h5f:
    flow = h5f['flow_fwd'][:].copy()
  flowVar = LocalFlowVar(flow)
  with h5py.File('test_flowVar_result.h5', 'w') as h5f:
    h5f.create_dataset('flowVar', data=flowVar)

  scale = (flowVar.min(), flowVar.max())
  flowVar = (flowVar - scale[0]) / (scale[1] - scale[0])
  im = Image.fromarray(flowVar * 255)
  im.show()