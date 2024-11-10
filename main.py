import os
from tools.FireImageOutliner import FireImageOutliner
from tools.FireDetection import FireDetection

def get_input(prompt: str) -> str:
    """Função auxiliar para obter entrada do usuário com espaço removido."""

    return input(prompt).strip().lower()

def process_single_image(fire_detection: FireDetection) -> None:
    """Processa uma única imagem para detecção de fogo."""

    img: str = get_input("Digite o caminho, nome e extensão do arquivo de imagem: ")
    fire_detection.test_if_image_has_fire(img)

def process_directory(fire_detection: FireDetection) -> None:
    """Processa todas as imagens em um diretório para detecção de fogo."""

    directory: str = get_input("Digite o caminho e o nome do diretório: ")

    if not os.path.isdir(directory):
        print("Diretório não encontrado. Tente novamente.")
        return loaded_model_loop(fire_detection)

    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)

        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"\nTestando imagem: {img_path}")
            fire_detection.test_if_image_has_fire(img_path)

def outline_image():
    """Exibe as outlines de uma imagem específica para verificar detecção de fogo."""

    img: str = get_input("Digite o caminho, nome e extensão do arquivo de imagem: ")
    image: FireImageOutliner = FireImageOutliner(img)
    print(f"A imagem tem outlines detectadas: {image.check_if_has_fire()}")
    image.show()

def loaded_model_loop(fire_detection: FireDetection):
    """Loop principal para testar imagens ou diretórios e exibir outlines."""

    test_option: str = get_input(
        'Deseja testar uma imagem ou um diretório com o modelo? '
        'Digite "img" para um arquivo, "dir" para um diretório, ou qualquer outra coisa para pular: '
    )

    if test_option == "img":
        process_single_image(fire_detection)
        loaded_model_loop(fire_detection)

    elif test_option == "dir":
        process_directory(fire_detection)
        loaded_model_loop(fire_detection)

    outline_option: str = get_input('Deseja olhar as outlines de uma imagem? Digite "y" para sim, qualquer outra coisa para sair: ')
    if outline_option == "y":
        outline_image()
    else:
        exit()

def main():
    fire_detection: FireDetection = FireDetection()

    option: str = get_input('Digite "criar" para criar novo modelo, ou "carregar" pra carregar um já existente: ')
    model_name: str = get_input("Digite o caminho e o nome do modelo: ")

    if option == "criar":
        dataset_name: str = get_input("Digite o caminho e o nome do dataset para treinar o modelo: ")
        epochs: int = int(get_input("Digite quantas épocas o modelo será treinado: "))
        fire_detection.initialize_model(model_name, dataset_name, epochs)

    elif option == "carregar":
        fire_detection.load_model(model_name)

    else:
        print("Opção inválida. Tente novamente.")
        main()

    summary_option: str = get_input('Deseja ver um resumo do modelo? Digite "y" para sim, qualquer outra coisa para não: ')

    if summary_option == "y":
        fire_detection.get_model_summary()

    loaded_model_loop(fire_detection)

if __name__ == '__main__':
    main()
