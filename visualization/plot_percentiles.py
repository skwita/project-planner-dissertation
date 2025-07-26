import matplotlib.pyplot as plt

def plot_percentile_analysis(df, output_image_path=None):
    _, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df["Процентиль"], df["Среднее время проекта"], label="Среднее время проекта", color="blue")

    for col in df.columns:
        if col.startswith("Простой_"):
            ax1.plot(df["Процентиль"], df[col], label=col)

    ax1.set_xlabel("Процентиль")
    ax1.set_ylabel("Время (дни)")
    ax1.legend()
    ax1.grid(True)

    if output_image_path:
        plt.savefig(output_image_path, bbox_inches='tight')
    plt.close()
