from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

from .utils import load_all_burgers
from .product import BurgerProduct
from .consumer import ConsumerAgent

class DiffusionModel(Model):
    def __init__(self, num_consumers: int, products_path: str):
        super().__init__()
        self.num_consumers = num_consumers
        # carregar produtos
        burger_defs = load_all_burgers(products_path)
        # converter em instâncias BurgerProduct
        self.burgers = {bid: BurgerProduct(bdata) for bid, bdata in burger_defs.items()}
        # scheduler / ativação
        self.schedule = RandomActivation(self)
        # criar agentes consumidores
        for i in range(num_consumers):
            # gerar preferências aleatórias simples
            prefs = {
                "is_vegan_pref": self.random.random() < 0.2,
                "likes_spicy": self.random.random() < 0.7,
                "intolerant_to_lactose": self.random.random() < 0.3
            }
            agent = ConsumerAgent(i, self, **prefs)
            self.schedule.add(agent)
        # (Opcional) definir vizinhanças / rede social
        # por simplicidade, cada agente escolhe alguns vizinhos aleatórios
        all_ids = list(range(num_consumers))
        for agent in self.schedule.agents:
            # por exemplo, definir 5 vizinhos aleatórios
            neighbors = self.random.sample(all_ids, k=min(5, num_consumers))
            # remover ele mesmo, se presente
            neighbors = [nid for nid in neighbors if nid != agent.unique_id]
            agent.neighbors = neighbors
        
        # coletor de dados
        self.datacollector = DataCollector(
            model_reporters={
                "num_tried": lambda m: sum(1 for a in m.schedule.agents if a.has_tried),
                # para cada burger, quantos agentes escolheram
                **{f"count_{bid}": (lambda m, b=bid: sum(1 for a in m.schedule.agents if a.current_choice == b))
                   for bid in self.burgers.keys()}
            }
        )
    
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
    
    def run_model(self, n_steps: int):
        for _ in range(n_steps):
            self.step()
