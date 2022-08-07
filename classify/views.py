from django.shortcuts import render
from django.http import HttpResponse
from django.db.models import Q
from models.combined_model import classify
from django.shortcuts import redirect
from django.views.decorators.csrf import csrf_exempt
import jsonlines

@csrf_exempt
def index(request):
	return render(request, 'index.html', {})

@csrf_exempt
def results(request):
	context = {}
	if (request.method=='GET'):
		text = request.GET.get("q")
		res, probs = classify(text)
		# res = 1
		# probs = [0.3,0.6,0.1]

		if (res==1):
			var = "True"
		elif res==2:
			var = "False"
		else:
			var = "Unclassified"
		context = {'var': var, 'query' : text, 'pr1' : probs[0], 'pr2' : probs[1], 'pr3' : probs[2]}
		return render(request, 'results.html', context)
	return render(request, 'results.html', context)

@csrf_exempt
def thank(request):
	context = {}
	if (request.method=='GET'):
		text = request.GET.get("query")
		res = request.GET.get("res")
		usr = request.GET.get("feedback")
		
		if (usr=="YES"): #store results
			new_data = {'text' : text, 'label' : res}
			with jsonlines.open('new_data.jsonl', mode='a') as writer:
				writer.write(new_data)

		context = {'var': res, 'query': text, 'feedback': usr}
		return render(request, 'thank.html', context)
	return render(request, 'thank.html', context)