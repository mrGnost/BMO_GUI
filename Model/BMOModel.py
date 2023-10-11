import numpy as np
import scipy.stats as stats
from scipy.constants import e, k
import warnings

warnings.filterwarnings("ignore")


def prepare_gene(x, args):
    data = args['const_values'].copy()
    # data[args['gene_idx']] = x
    data[int(args['gene_idx'][0])] = x[0]
    data[int(args['gene_idx'][1])] = x[1]
    return data


def square_func(x, args):
    data = prepare_gene(x, args)
    return np.sum(data ** 2)


def complex_func(x, args):
    data = prepare_gene(x, args)
    return np.sum(-data * np.sin(np.sqrt(np.abs(data))))


def multimodal_func(x, args):
    data = prepare_gene(x, args)
    return (data[1] - 5.1 / (4 * np.pi ** 2) * data[0] ** 2 + 5 / np.pi * data[0] - 6) ** 2 + \
           10 * (1 - 1 / (8 * np.pi) * np.cos(data[0]) + 10)

def double_diode(x, args):
    return np.mean((args['I_t'] - x[2]
                    + x[3] * (np.exp(e * (args['V_t'] + x[0] * args['I_t']) / (x[5] * k * args['T'])) - 1)
                    + x[4] * (np.exp(e * (args['V_t'] + x[0] * args['I_t']) / (x[6] * k * args['T'])) - 1)
                    + (args['V_t'] + x[0] * args['I_t']) / x[1]) ** 2)


def single_diode(x, args):
    return np.mean((args['I_t'] - x[2]
                    + x[3] * (np.exp(e * (args['V_t'] + x[0] * args['I_t']) / (x[4] * k * args['T'])) - 1)
                    + (args['V_t'] + x[0] * args['I_t']) / x[1]) ** 2)


functions = {
    'y=x^2': square_func,
    'y=-x*sin(sqrt(abs(x)))': complex_func,
    'multimodal function': multimodal_func
}


class Bird:
    def __init__(self, society, id):
        self.society = society
        self.id = id
        self.gene = np.random.rand(self.society.gene_size) * (
                self.society.gene_max - self.society.gene_min) + self.society.gene_min

    def get_score(self):
        return self.society.objective_func(self.gene, self.society.func_args)

    def mate(self, birds=None):
        if self.strategy in ['monogamy', 'promiscuity']:
            self.child_gene = self.gene + self.society.w * np.random.rand(self.society.gene_size) * (
                    birds.gene - self.gene)
            self.make_mutation()

        elif self.strategy in ['polygyny', 'polyandry']:
            diff = np.zeros(self.society.gene_size)
            for i in range(len(birds)):
                diff += np.random.rand(self.society.gene_size) * (birds[i].gene - self.gene)
            self.child_gene = self.gene + self.society.w * diff
            self.make_mutation()

        elif self.strategy == 'parthenogenesis':
            self.child_gene = np.zeros(self.society.gene_size)
            for i in range(self.society.gene_size):
                if self.rand_norm(1)[0] > self.society.mcfp:
                    self.child_gene[i] = self.gene[i] + self.society.m * (self.rand_norm(1)[0] - self.rand_norm(1)[0])
                else:
                    self.child_gene[i] = self.gene[i]

        self.child_gene = np.maximum(np.minimum(self.child_gene, self.society.gene_max), self.society.gene_min)

    def make_mutation(self):
        if self.society.mw == None:
            c = np.random.randint(self.society.gene_size)
            if self.rand_norm(1)[0] > self.society.mcf:
                self.child_gene[c] = self.society.gene_min[c] - self.rand_norm(1)[0] * (
                        self.society.gene_min[c] - self.society.gene_max[c])
        else:
            c = self.rand_norm(self.gene.shape[0]) >= self.society.mcf
            self.child_gene[c] = self.gene[c] + self.society.mw * (self.rand_norm(1)[0] - self.rand_norm(1)[0]) * (
                    self.society.gene_max[c] - self.society.gene_min[c])

    def set_strategy(self, strategy):
        self.strategy = strategy
        if self.strategy in ['parthenogenesis', 'polyandry']:
            self.sex = 'female'
        else:
            self.sex = 'male'

    def make_promiscuous(self):
        self.strategy = 'promiscuity'
        self.gene = np.array(
            self.society.chaotic_next() * (self.society.gene_max - self.society.gene_min) + self.society.gene_min)
        return self

    def to_next_generation(self):
        if self.society.objective_func(self.child_gene, self.society.func_args) < self.get_score():
            self.gene = self.child_gene

    def rand_norm(self, n):
        return stats.truncnorm.rvs(-5, 5, loc=0.5, scale=0.1, size=n)

    def __repr__(self):
        return "gene: {}, score: {}".format(self.gene, self.get_score())


class Society:
    def __init__(
            self,
            society_size=100,
            gene_size=10,
            gene_min=np.array([-1000] * 10),
            gene_max=np.array([1000] * 10),
            seed=None,
            parts=np.array([5, 5, 50, 30]),
            generations_max=100,
            poly_mates=3,
            mono_top=10,
            mcf=0.9,
            w_start=2,
            w_fin=0,
            T_start=None,
            T_fin=None,
            mw_start=None,
            mw_fin=None,
            m=1e-2,
            objective_func=square_func,
            func_args=None,
            enable_log=False,
            log_interval=1,
            log_birds_count=5
    ):
        self.society_size = society_size
        self.gene_size = gene_size
        self.gene_min = gene_min
        self.gene_max = gene_max
        self.seed = seed
        if self.seed != None:
            np.random.seed(self.seed)
        self.split = np.cumsum(parts, dtype=int)
        self.generations_max = generations_max
        self.poly_mates = poly_mates
        self.mono_top = mono_top
        self.mcf = mcf
        self.mcfp = 1 / (self.generations_max + 1)
        self.mcfp_step = 1 / (self.generations_max + 1)
        self.w_start = w_start
        self.w = self.w_start
        self.w_step = (w_start - w_fin) / self.generations_max
        self.T_start = T_start
        self.T = self.T_start
        self.enable_log = enable_log
        self.log_interval = log_interval
        self.log_birds_count = log_birds_count
        if T_start != None:
            self.T_step = (T_start - T_fin) / self.generations_max
        self.mw_start = mw_start
        self.mw = self.mw_start
        if mw_start != None:
            self.mw_step = (mw_start - mw_fin) / self.generations_max
        self.m = m
        self.generation = None
        self.chaotic_number = np.random.rand()
        self.has_parth = parts.shape[0] == 4
        self.objective_func = functions[objective_func]
        self.func_args = func_args

    def train(self):
        self.chaotic_number = np.random.rand()
        self.generation = np.empty(self.society_size, dtype=object)
        for i in range(self.society_size):
            self.generation[i] = Bird(self, i)
        self.mcfp = 1 / (self.generations_max + 1)
        self.w = self.w_start
        self.T = self.T_start
        self.mw = self.mw_start
        indices = np.arange(self.society_size)
        parth = None
        if self.has_parth:
            parth, poly_f, mono, poly_m, prom = np.split(indices, self.split)
            fem = np.concatenate((parth, poly_f), axis=None)
        else:
            poly_f, mono, poly_m, prom = np.split(indices, self.split)
            fem = poly_f.copy()
        male = mono[:self.mono_top]

        return parth, poly_f, poly_m, mono, prom, male, fem
        #
        # while gen < self.generations_max:
        #     self.make_step(parth, poly_f, poly_m, mono, prom, male, fem, gen)

    def make_step(self, parth, poly_f, poly_m, mono, prom, male, fem):
        sorted_gen = list(self.generation)
        sorted_gen.sort(key=lambda x: x.get_score())
        self.generation = np.array(sorted_gen)

        if self.has_parth:
            for bird in self.generation[parth]:
                bird.set_strategy('parthenogenesis')
        for bird in self.generation[poly_f]:
            bird.set_strategy('polyandry')
        for bird in self.generation[mono]:
            bird.set_strategy('monogamy')
        for bird in self.generation[poly_m]:
            bird.set_strategy('polygyny')
        for bird in self.generation[prom]:
            bird.make_promiscuous()

        scores_male = np.zeros(male.shape[0], dtype=float)
        scores_fem = np.zeros(fem.shape[0], dtype=float)
        probas_male = np.zeros(male.shape[0], dtype=float)
        probas_fem = np.zeros(fem.shape[0], dtype=float)
        for i in range(male.shape[0]):
            scores_male[i] = self.generation[male[i]].get_score()
            probas_male[i] = scores_male[i]
        for i in range(fem.shape[0]):
            scores_fem[i] = self.generation[fem[i]].get_score()
            probas_fem[i] = scores_fem[i]
        probas_fem = 1 / probas_fem / ((1 / probas_fem).sum())
        probas_male = 1 / probas_male / ((1 / probas_male).sum())

        if self.has_parth:
            for bird in self.generation[parth]:
                bird.mate()

        for bird in self.generation[poly_f]:
            p = probas_male if self.T is None else self.get_poly_probas(bird, scores_male)
            if self.T is None:
                bird.mate(np.random.choice(self.generation[male], self.poly_mates, p=probas_male, replace=False))
            else:
                self.poly_mate(bird, self.generation[male], scores_male)
        for bird in self.generation[mono]:
            bird.mate(np.random.choice(self.generation[fem], p=probas_fem))
        for bird in self.generation[poly_m]:
            if self.T is None:
                bird.mate(np.random.choice(self.generation[fem], self.poly_mates, p=probas_fem, replace=False))
            else:
                self.poly_mate(bird, self.generation[fem], scores_fem)
        for bird in self.generation[prom]:
            bird.mate(np.random.choice(self.generation[male], p=probas_male))

        self.prev_gen_genes = np.stack([self.generation[i].gene for i in range(self.society_size)], axis=0)
        for bird in self.generation:
            bird.to_next_generation()
        self.new_gen_genes = np.stack([self.generation[i].gene for i in range(self.society_size)], axis=0)
        self.birds_idx = np.array([self.generation[i].id for i in range(self.society_size)])

        self.mcfp += self.mcfp_step
        self.w -= self.w_step
        if self.T is not None:
            self.T -= self.T_step
        if self.mw is not None:
            self.mw -= self.mw_step

    def chaotic_next(self):
        self.chaotic_number = 4 * self.chaotic_number * (1 - self.chaotic_number)
        return self.chaotic_number

    def get_poly_probas(self, x, birds_scores):
        return np.exp(-np.abs(birds_scores - x.get_score()) / self.T)

    def poly_mate(self, x, birds, scores):
        x.mate(birds[np.random.rand(birds.shape[0]) < self.get_poly_probas(x, scores)])

    def show_to_log(self, gen_num):
        print("GENERATION #", gen_num, sep='')
        for bird in self.generation[:self.log_birds_count]:
            print(bird)
        print('------------------------------------------------\n')

    def get_last_gen(self):
        return self.generation

    def get_best(self):
        sorted_gen = list(self.generation)
        return min(sorted_gen, key=lambda x: x.get_score())


def get_experimental_values(path):
    if path == "":
        return None
    with open(path, 'r') as file:
        lines = file.readlines()
        if len(lines) == 0:
            return None
        param_names = lines[0][:-1].split('\t')
        if len(param_names) == 0 or param_names[0] == '':
            return None
        result = {}
        for name in param_names:
            result[name] = np.array([])
        for i in range(1, len(lines)):
            vals = list(map(float, lines[i][:-1].split()))
            for j in range(len(vals)):
                result[param_names[j]] = np.append(result[param_names[j]], [vals[j]], axis=0)
        return result
