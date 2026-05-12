class ServiceException(Exception):
    """自定义服务异常"""
    
    def __init__(
        self,
        message: str,
    ):
        super().__init__(message)
        self.message = message
