from tools.FireImageOutliner import FireImageOutliner
from tools.FireDetection import FireDetection

def main():
    fireDetection: FireDetection = FireDetection()

    # fireDetection.initialize_model("models/modelo_deteccao_fogo", "dataset/fire_dataset", 16)
    fireDetection.load_model("models/modelo_deteccao_fogo")

    fireDetection.get_model_summary()

    for i in range(5,15):
        img_path = 'dataset/fire_dataset/fire/'+str(i)+'.jpg'
        fireDetection.test_if_image_has_fire(img_path)

    for i in range(1,11):
        img_path = 'dataset/fire_dataset/not_fire/'+str(i)+'.jpg'
        fireDetection.test_if_image_has_fire(img_path)

    image: FireImageOutliner = FireImageOutliner("dataset/fire_dataset/fire/6.jpg")
    print(image.check_if_has_fire())
    image.show()

if __name__ == '__main__':
    main()
