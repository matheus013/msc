class BurgerProduct:
    def __init__(self, info_dict: dict):
        # info_dict vem do YAML carregado
        self.id = info_dict["id"]
        self.name = info_dict.get("name")
        self.price = info_dict.get("price")
        self.ingredients = info_dict.get("ingredients", [])
        self.weight_g = info_dict.get("weight_g")
        self.is_vegan = info_dict.get("is_vegan", False)
        self.is_vegetarian = info_dict.get("is_vegetarian", False)
        self.is_spicy = info_dict.get("is_spicy", False)
        self.contains_lactose = info_dict.get("contains_lactose", False)
        self.calories = info_dict.get("calories")
        self.protein_g = info_dict.get("protein_g")
    
    def utility_for(self, consumer):
        """
        Compute a “utility / attractiveness” deste produto para um agente consumidor.
        Por exemplo, o consumidor pode ter preferências por veganismo, tolerância ao picante etc.
        Retornar um valor numérico de utilidade que pode ser usado em decisões probabilísticas.
        """
        # exemplo simples: base utility = protein / price, penalidades, bônus
        util = self.protein_g / (self.price + 1e-6)
        # penalidade se produto contém lactose mas consumidor é intolerante
        if consumer.intolerant_to_lactose and self.contains_lactose:
            util *= 0.5
        # se consumidor é vegano mas produto não é vegano, penalidade forte
        if consumer.is_vegan_pref and not self.is_vegan:
            util *= 0.2
        # se consumidor não gosta de picante e produto é picante
        if not consumer.likes_spicy and self.is_spicy:
            util *= 0.7
        # você pode somar mais fatores (ingredientes preferidos, familiaridade, etc)
        return util
