from dataclasses import dataclass
import numpy as np
@dataclass(frozen=True)
class Formula:
    a:str; op:str; b:str; sa:float=1.; sb:float=1.
    def expression(self): return f'({self.sa:g}*{self.a}) {self.op} ({self.sb:g}*{self.b})'
def generate_formulas(features,budget=1_000_000,seed=7):
    rng=np.random.default_rng(seed); ops=['+','-','*']
    for _ in range(budget):
        a,b=rng.choice(features,2); yield Formula(str(a),str(rng.choice(ops)),str(b),float(rng.choice([-1,-.5,.5,1,2])),float(rng.choice([-1,-.5,.5,1,2])))
def eval_formula(f,frame):
    a=frame[f.a].to_numpy(float)*f.sa; b=frame[f.b].to_numpy(float)*f.sb
    return a+b if f.op=='+' else a-b if f.op=='-' else a*b
