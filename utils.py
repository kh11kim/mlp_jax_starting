
class Hyperparam:
    def __init__(self):
        pass

    def __contains__(self, key):
        return key in self.__dict__.keys()
    
    def __repr__(self):
        return self.__dict__.__repr__()
    
    def to_str(self):
        result = ""
        for key, item in self.__dict__.items():
            if key == "layers":
                result += key + ":" + "_".join([str(l) for l in item])
            else:
                result += key + ":" + str(item)
            result += ","
        return result[:-1]