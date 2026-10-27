Source: https://github.com/fkryan/gazelle
Revision: 2d78f9f3bd2a5db360354954ec1a4c526ee8ab55

backbone.py, model.py and utils.py are unmodified upstream files.
An __init__.py was added because upstream find_packages omits this directory.
The application supplies a pinned DINOv2 backbone instead of the upstream unpinned loader.
MIT license retained in LICENSE. Weights are downloaded separately.
