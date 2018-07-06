from django.http import HttpResponse
from django.template import loader
from django.views.decorators.cache import cache_control

from conf import conf

# the page doesn't get refreshed often enough to legitimize 
# long caching.
@cache_control(max_age=3)
def index(request):
    template = loader.get_template('sentiment/index.html')
    context = {'analyze_comments_endpoint': conf['chatty_rest']['fqdn'] + 'chatty/'}
    return HttpResponse(template.render(context, request))
