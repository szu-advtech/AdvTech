import json

class MovementPathGenerator:

    def __init__(self) -> None:

        self.rotXfunc = lambda t: 0
        self.rotYfunc = lambda t: 0
        self.rotZfunc = lambda t: 0

        self.transXfunc = lambda t: 0
        self.transYfunc = lambda t: 0
        self.transZfunc = lambda t: -4800 + t * 2

    
    def generate(self, outname="./zoom.json") -> None:
        
        path = []
        for t in range(600):
            d = {
                "rotation": [
                    self.rotXfunc(t),
                    self.rotYfunc(t),
                    self.rotZfunc(t),
                ],
                "translation": [
                    self.transXfunc(t),
                    self.transYfunc(t),
                    self.transZfunc(t),
                ]
            }
            path.append(d)
        
        f = open(outname, "w")
        json.dump(path, f)


if __name__ == "__main__":
    MPG = MovementPathGenerator()
    MPG.generate()