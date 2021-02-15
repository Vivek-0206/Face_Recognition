# import tools
from face_Recognition import create_model, predict_person

print("_____Start loading model_____")
model = create_model()

predict_person(model=model)
