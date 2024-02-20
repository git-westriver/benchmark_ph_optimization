import torch

class Regularization:
    """
    Base class for regularization.
    """
    def __call__(self, X, *args, **kwargs) -> torch.Tensor:
        """
        Compute the regularization term.

        Parameters:
            - X(torch.Tensor): point cloud. shape=(# of points, dim)

        Returns:
            - loss(torch.Tensor): the regularization term.
        """
        raise NotImplementedError
    
    def projection(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project the point cloud to the feasible set.

        Parameters:
            - X(torch.Tensor): point cloud. shape=(# of points, dim)

        Returns:
            - X(torch.Tensor): projected point cloud.
        """
        raise NotImplementedError

class RectangleRegularization(Regularization):
    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float, lamb: float, order: int):
        self.lamb = lamb
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
    
    def __call__(self, X):
        penalty_x = torch.relu(X[:, 0] - self.x_max) + torch.relu(self.x_min - X[:, 0])
        penalty_y = torch.relu(X[:, 1] - self.y_max) + torch.relu(self.y_min - X[:, 1])
        return self.lamb * (torch.sum(penalty_x ** self.order) + torch.sum(penalty_y ** self.order))
    
    def projection(self, X):
        X[:, 0] = torch.clamp(X[:, 0], self.x_min, self.x_max)
        X[:, 1] = torch.clamp(X[:, 1], self.y_min, self.y_max)
        return X
