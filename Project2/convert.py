from pydub import AudioSegment
import glob
import os

def convert(mp3_path, upper_path):
    sound = AudioSegment.from_mp3(mp3_path)
    file_name = mp3_path.split('/')[-1].split('.')[0]
    directory = os.path.dirname("music_wav/"+upper_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    sound.export(directory + '/'+ file_name+'.wav', format="wav")

if __name__ == "__main__":
    path = glob.glob('music_dataset/*')
    for f_path in path:
        upper_path = f_path.split('/')[-1]
        music_path = glob.glob(f_path+'/*')
        for mp3 in music_path:
            convert(mp3, upper_path)
        

