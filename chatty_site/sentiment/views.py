from django.http import HttpResponse
from django.template import loader
from django.views.decorators.cache import cache_control

# the page doesn't get refreshed often enough to legitimize 
# long caching.
@cache_control(max_age=3)
def index(request):
    template = loader.get_template('sentiment/index.html')
    context = {}
    return HttpResponse(template.render(context, request))
