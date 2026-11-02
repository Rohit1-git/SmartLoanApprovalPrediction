from django import template

register = template.Library()


@register.filter
def get_item(dictionary, key):
    """Return dictionary[key] safely in a Django template."""
    if dictionary is None:
        return ""
    try:
        return dictionary.get(key, "")
    except AttributeError:
        return ""
