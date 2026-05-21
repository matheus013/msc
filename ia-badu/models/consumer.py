from mesa import Agent

class ConsumerAgent(Agent):
    def __init__(self, unique_id, model, **prefs):
        super().__init__(unique_id, model)
        # prefs pode incluir preferências: vegan, tolerância lactose, gosto por picante etc.
        self.is_vegan_pref = prefs.get("is_vegan_pref", False)
        self.likes_spicy = prefs.get("likes_spicy", True)
        self.intolerant_to_lactose = prefs.get("intolerant_to_lactose", False)
        # estado de adoção
        self.has_tried = False
        self.current_choice = None  # id do burger escolhido / adotado
        self.neighbors = []  # lista de IDs de vizinhos (ou agentes reais)
    
    def step(self):
        # Processo de decisão em cada iteração
        # Por exemplo: se ainda não provou, decide experimentar um burger baseado em utilidades + influência
        if not self.has_tried:
            # exemplo: escolhe probabilisticamente um burger do catálogo
            # influência dos vizinhos pode aumentar utilidade percebida de produtos que eles adotaram
            # obter utilidades para cada produto
            burgers = self.model.burgers  # dict id → BurgerProduct
            utilities = {}
            for bid, burger in burgers.items():
                base_u = burger.utility_for(self)
                # influência social: se vizinho já adotou esse burger, adicionar bônus
                num_neighbors_adopted = sum(1 for nb in self.neighbors
                                            if self.model.schedule._agents[nb].current_choice == bid)
                # por exemplo, cada vizinho adotante dá +0.1 multiplicativo
                u = base_u * (1 + 0.1 * num_neighbors_adopted)
                utilities[bid] = u
            # normalizar probabilidades
            total = sum(utilities.values())
            if total > 0:
                import random
                # escolher com prob proporcional à utilidade
                rnd = random.random() * total
                cum = 0
                for bid, u in utilities.items():
                    cum += u
                    if rnd <= cum:
                        choice = bid
                        break
                # adotar / experimentar
                self.current_choice = choice
                self.has_tried = True
        else:
            # se já experimentou, pode decidir continuar ou desistir em iterações futuras — opcional
            pass
