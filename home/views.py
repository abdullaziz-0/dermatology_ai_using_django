from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_protect
from django.core.files.storage import FileSystemStorage
from ultralytics import YOLO


MODEL_PATH_SKIN = r"C:\Users\Abdul\Downloads\train4(Skin&NotSKin)\train4(Skin&NotSKin)\weights\best.pt"
MODEL_PATH_BURN = r"C:\xampp\htdocs\skin burn\train2(yolov8L)\weights\best.pt"

@csrf_protect
def home(request):
    if request.method == 'POST' and request.FILES.get('burn-image'):
        uploaded_file = request.FILES['burn-image']
        fs = FileSystemStorage()  
        temp_file_name = fs.save(uploaded_file.name, uploaded_file)
        temp_file_path = fs.path(temp_file_name) 
        temp_file_url = fs.url(temp_file_name)  

        try:
            skin_NonSkin_model = YOLO(MODEL_PATH_SKIN, task='detect')
            res = skin_NonSkin_model(temp_file_path) 
            for r in res:
                maxprop = r.probs.top1
                if maxprop == 1:  
                    model = YOLO(MODEL_PATH_BURN, task='detect')
                    result = model(temp_file_path)  
                    for r in result:
                        maxprop = r.probs.top1
                        if maxprop == 0:
                            result = "First-Degree Burn"
                        elif maxprop == 1:
                            result = "Second-Degree Burn"
                        elif maxprop == 2:
                            result = "Third-Degree Burn"
                        else:
                            result = "Normal skin"
                    break
                else:
                    result = "Not human skin"
        except Exception as e:
            result = str(e)
        
        return JsonResponse({'result': result, 'image_url': temp_file_url})

    return render(request, "api/home.html")
