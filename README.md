## Compilation

### Sur unr machine de l'école

**1.** S'authentifier sur une machine de l'école :

```sh
# Se connecter à une machine et s'autentifier
ssh -X -p 2200[0-4] login@gpgpu.image.lrde.iaas.epita.fr
kinit login
aklog

# Créer un dossier pour le projet à monter (NE FAIRE QU'UNE FOIS)
cd afs/
mkdir gpgpu-project
```

**2.** Sur un autre terminal en local, monter le dossier du projet :

```sh
# Monter le repo git sur l'afs
sshfs -o allow_other -p 2200[0-4] login@gpgpu.image.lrde.iaas.epita.fr:/path/to/afs/gpgpu-project path/to/git/project
```

**3.** Reprendre le terminal de l'étape 1 et se placer dans le dossier monté :

```sh
# Se mettre dans le dossier monté
cd gpgpu-project

# Creer un nouveau shell avec toutes les configuration et dépendances du projet
nix-shell

# Build le projet
./build.sh
```

### En local avec sa propre carte Nvidia

```sh
docker build -t gpgpu-image .
docker run --name=gpgpu -it --rm --gpus=all -v $(pwd):/gpgpu gpgpu-image

# Dans le container build le projet
./build.sh
```

### Description du build

1. Installer gstreamer (voir le fichier nix-shell associé)
2. Builder le code d'exemple avec cmake (le filtre se trouve dans le dossier `./build`)
3. Télécharger la vidéo d'exemple `https://gstreamer.freedesktop.org/media/sintel_trailer-480p.webm`
4. Exporter le chemin du filtre dans la variable d'environnement `GST_PLUGIN_PATH`
5. Ajouter un symlink vers le plugin C++ ou sa version CUDA
6. Lancer l'application du filter sur la vidéo et l'enregistrer en mp4 _dans votre afs_
7. En local, visualiser la vidéo avec _vlc_

## Code

Les seuls fichiers à modifier sont normalement `filter_impl.cu` (version cuda) et `filter_impl.cpp` (version cpp). Pour basculer entre l'utilisation du filter en C++ et du filtre en CUDA, changer le lien symbolique vers le bon `.so`.

## Uiliser _gstreamer_

### Flux depuis la webcam -> display

Si vous avez une webcam, vous pouvez lancer gstreamer pour appliquer le filter en live et afficher son FPS.

```sh
gst-launch-1.0 -e -v v4l2src ! jpegdec ! videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! fpsdisplaysink
```

### Flux depuis une vidéo locale -> display

Même chose pour une vidéo en locale.

```sh
gst-launch-1.0 -e -v uridecodebin uri=file://$(pwd)/sintel_trailer-480p.webm !  videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! fpsdisplaysink
```

### Flux depuis une vidéo locale -> vidéo locale

Pour sauvegarder le résulat de l'application de votre filtre.

```sh
gst-launch-1.0 uridecodebin uri=file://$(pwd)/sintel_trailer-480p.webm ! videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! video/x-raw, format=I420 ! x264enc ! mp4mux ! filesink location=video.mp4
```

### Bench FPS du traitement d'une vidéo

Enfin pour bencher la vitesse de votre filtre. Regarder la sortie de la console pour voir les fps.

```sh
gst-launch-1.0 -e -v uridecodebin uri=file://$(pwd)/sintel_trailer-480p.webm !  videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! fpsdisplaysink video-sink=fakesink sync=false
```
