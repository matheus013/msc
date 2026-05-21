from models.diffusion_model import DiffusionModel
import matplotlib.pyplot as plt

def run_one():
    model = DiffusionModel(num_consumers=100, products_path="data/products")
    model.run_model(n_steps=20)
    df = model.datacollector.get_model_vars_dataframe()
    print(df)
    # plot resultados
    df.plot(kind="line", figsize=(10, 6))
    plt.xlabel("Step")
    plt.ylabel("Count / Tried")
    plt.title("Diffusion of Burger Products")
    plt.show()

if __name__ == "__main__":
    run_one()
