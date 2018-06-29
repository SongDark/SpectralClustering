from sklearn import datasets

def genTwoCircles(n_samples=1000):
    X,y = datasets.make_circles(n_samples, factor=0.5, noise=0.05)
    return X, y

