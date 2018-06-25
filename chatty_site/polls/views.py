# from django.shortcuts import render

# # Create your views here.

from django.http import HttpResponse


def index(request):
    return HttpResponse("Hello pollster")


def detail(request, question_id):
    return HttpResponse("You're look at question %s." % question_id)


def results(request, question_id):
    response = "You're look at the results of a question %s."
    return HttpResponse(response % question_id)


def vote(request, question_id):
    return HttpResponse("You're voting on question %s." % question_id)
