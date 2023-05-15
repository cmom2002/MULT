import warnings
import librosa
import librosa.display
import librosa.beat
import numpy as np
import os
import scipy.stats as st
from scipy.spatial import distance

POS = 0

FNAME = "MT0000202045"

#------------------- Exercicio 2 ------------------------
#-----2.1-----
def normalize(array, save_file_path):
   
    features = array

    columns = array.shape[1]
    
    features = features[0:, 0:columns]
    for i in range(columns):
        
        maxi = features[:, i].max()
        mini = features[:, i].min()
        if maxi == mini:
            features[:, i] = 0
        else:
            features[:, i] = (features[:, i] - mini)/(maxi - mini)

    np.savetxt(save_file_path, features, fmt="%f", delimiter=",")
    
    return features

#-----2.2-----
def extract_all(file_path):
    files = sorted(os.listdir(file_path))

    song_names = files[1:, 0]

    feature = np.empty((len(files), 190))

    sr = 22050

    for i in range(len(files)):
        
        file_name = file_path + "/" + files[i]
        y = librosa.load(file_name, sr=sr, mono=True)[0]

        mfcc = librosa.feature.mfcc(y=y, n_mfcc=13)
        feature[i, : 91] = stats_feature(mfcc)

        centroid = librosa.feature.spectral_centroid(y=y)
        feature[i, 91: 98] = stats_feature(centroid)

        bw = librosa.feature.spectral_bandwidth(y=y)
        feature[i, 98: 105] = stats_feature(bw)

        contrast = librosa.feature.spectral_contrast(y=y)
        feature[i, 105: 154] = stats_feature(contrast)

        flatness = librosa.feature.spectral_flatness(y=y)
        feature[i, 154: 161] = stats_feature(flatness)

        rolloff = librosa.feature.spectral_rolloff(y=y)
        feature[i, 161: 168] = stats_feature(rolloff)

        f0 = librosa.yin(y=y, fmin=20, fmax=sr/2)
        f0[f0 == sr/2] = 0
        feature[i, 168: 175] = stats_feature(f0)

        rms = librosa.feature.rms(y=y)
        feature[i, 175: 182] = stats_feature(rms)

        zcr = librosa.feature.zero_crossing_rate(y=y)
        feature[i, 182: 189] = stats_feature(zcr)

        tempo = librosa.beat.tempo(y=y)
        feature[i, 189] = tempo
    
    np.savetxt("Results/Exercise2/Exercise_2.2_not_normalized.csv", feature, fmt="%f", delimiter=",")
    
    if not os.path.exists("Results/song_names.txt"):
        np.savetxt("Results/song_names.txt", song_names, delimiter="\n")
     
    return feature

        

def stats_feature(feature):
    axis = 0
    if feature.ndim > 1:
        axis = 1
    mean = np.mean(feature, axis=axis)
    std = np.std(feature, axis=axis)
    skew = st.skew(feature, axis=axis)
    kurt = st.kurtosis(feature, axis=axis)
    med = np.median(feature, axis=axis)
    maxi = np.max(feature, axis=axis)
    mini = np.min(feature, axis=axis)

    if axis == 0:
        final = np.array([mean, std, skew, kurt, med, maxi, mini])
    else: 
        lines = feature.shape[0]
        final = np.empty((lines, 7))
        for i in range(lines):
            final[i, :] = [mean[i], std[i], skew[i], kurt[i], med[i], maxi[i], mini[i]]
        
    final = final.flatten()
    return final


#------------------- Exercicio 3 ------------------------
#-----3.2-----
def metrics_and_similarity(array, name):
    euclidean = np.zeros((900, 900))
    manhattan = np.zeros((900, 900))
    cosine = np.zeros((900, 900))

    if(os.path.exists("Results/Exercise3.2/Exercise_3.2_cosine_" + name + ".csv")):
        print("JA EXISTE 3.2 " + name)
        euclidean = np.genfromtxt("Results/Exercise3.2/Exercise_3.2_euclidean_" + name + ".csv", delimiter=',', dtype=float)
        manhattan = np.genfromtxt("Results/Exercise3.2/Exercise_3.2_manhattan_" + name + ".csv", delimiter=',', dtype=float)
        cosine = np.genfromtxt("Results/Exercise3.2/Exercise_3.2_cosine_" + name + ".csv", delimiter=',', dtype=float)

    else:
        feature = array
        feature = feature[ :, 1:(feature.shape[1])]
    
        for i in range(900):
            for j in range((i + 1), 900):
                euclidean[i][j] = distance.euclidean(feature[i, : ], feature[j, : ])
                euclidean[j][i] = euclidean[i][j]
                manhattan[i][j] = distance.cityblock(feature[i, : ], feature[j, : ])
                manhattan[j][i] = manhattan[i][j]
                cosine[i][j] = distance.cosine(feature[i, : ], feature[j, : ])
                cosine[j][i] = cosine[i][j]

        np.savetxt("Results/Exercise3.2/Exercise_3.2_euclidean_" + name + ".csv", euclidean, fmt="%f", delimiter=",")
        np.savetxt("Results/Exercise3.2/Exercise_3.2_manhattan_" + name + ".csv", manhattan, fmt="%f", delimiter=",")
        np.savetxt("Results/Exercise3.2/Exercise_3.2_cosine_" + name + ".csv", cosine, fmt="%f", delimiter=",")

    return [euclidean, manhattan, cosine]

def metrics_euclidean(array, name):
    euclidean = np.zeros((900, 900))

    feature = array
    feature = feature[ :, 1:(feature.shape[1])]

    for i in range(900):
        for j in range(i + 1, 900):
            euclidean[i][j] = distance.euclidean(feature[i,: ], feature[j,: ])
            euclidean[j][i] = euclidean[i][j]

    np.savetxt("Results/Exercise3.2/Exercise_3.2_euclidean_" + name + ".csv", euclidean, delimiter=',', fmt="%f")

    return euclidean

def metrics_manhattan(array, name):
    manhattan = np.zeros((900, 900))

    feature = array
    feature = feature[ :, 1:(feature.shape[1])]

    for i in range(900):
        for j in range(i + 1, 900):
            manhattan[i][j] = distance.cityblock(feature[i,: ], feature[j,: ])
            manhattan[j][i] = manhattan[i][j]

    np.savetxt("Results/Exercise3.2/Exercise_3.2_manhattan_" + name + ".csv", manhattan, delimiter=',', fmt="%f")

    return manhattan

def metrics_cosine(array, name):
    cosine = np.zeros((900, 900))

    feature = array
    feature = feature[ :, 1:(feature.shape[1])]

    for i in range(900):
        for j in range(i + 1, 900):
            cosine[i][j] = distance.cosine(feature[i,: ], feature[j,: ])
            cosine[j][i] = cosine[i][j]

    np.savetxt("Results/Exercise3.2/Exercise_3.2_cosine_" + name + ".csv", cosine, delimiter=',', fmt="%f")

    return cosine


#-----3.3-----
def similarity_rankings(array, name):
    rank = np.zeros((array.shape[0], 20))

    for i in range(array.shape[0]):
        rank[i,:] = np.argsort(array[i])[:20]

    np.savetxt("Results/Exercise3.3/Exercise_3.3_" + name + ".csv", rank, fmt="%f", delimiter=",")

    return rank

#------------------- Exercicio 4 ------------------------
#-----4.1.1-4.1.2-----
def song_scores(path):
    array = np.genfromtxt(path, delimiter=',', dtype=str)[1:,:]
    
    sim_data = np.zeros((array.shape[0], array.shape[0]))
    lines = array.shape[0]

    for i in range(lines):
        artist = array[i][1]
        quadrant = array[i][3]
        
        moods_strsplit = np.array(array[i][9][1:-1].split("; "))
        genres_str = np.array(array[i][11][1:-1].split("; "))

        for j in range(i, lines):
            ranking = 0
            if artist == array[j][1]:
                ranking += 1
            
            if quadrant == array[j][3]:
                ranking += 1
            
            ranking += len(np.intersect1d(moods_strsplit, np.array(array[j][9][1:-1].split("; "))))
            ranking += len(np.intersect1d(genres_str, np.array(array[j][11][1:-1].split("; "))))

            sim_data[i][j] = ranking
            sim_data[j][i] = ranking

    np.savetxt("Results/Exercise4/Exercise_4.1.2_similarity.csv", sim_data, fmt="%0.0f", delimiter=",")

    return sim_data

#-----4.1.3-----
def construct_string(array):
    aux = "["
    count = 1
    for i in array:
        aux += "\'" + i +".mp3\'"
        if(count % 3 == 0):
            if (count == 21):
                    aux += "]\n"
            else:
                aux += '\n'
        aux += ' '
        count += 1
    aux += '\n'
    return aux


def get_song_position(song_names, track):
    pos = 0
    for i in range(len(song_names)):
        if song_names[i] == track:
            pos = i
            break
    return pos


def ranking_distance(track, song_names, distance):
    music_pos = get_song_position(song_names, track)

    rank = np.argsort(distance[music_pos, :])
    rank = rank[1:21]
    
    array_songs = [track]
    for pos in rank:
        array_songs.append(song_names[pos])
    
    return rank, array_songs


def ranking_metadata(track, song_names, sim_data):
    music_pos = get_song_position(song_names, track)

    rank = np.argsort(sim_data[music_pos, :])[::-1]
    rank = rank[1:21]
    array_songs = [track]
    
    for pos in rank:
        if (song_names[pos] != track):
            array_songs.append(song_names[pos])
            
    return rank, array_songs


def score_metadata(song_names, track, array_distance, array_metadata, sim_data):
    score_meta = []
    
    music_pos = get_song_position(song_names, track)
    score_meta.append(sim_data[music_pos][music_pos])

    for i in range(len(array_metadata)):
        music_pos1 = get_song_position(song_names, array_distance[i])
        music_pos2 = get_song_position(song_names, array_metadata[i])
        score_meta.append(sim_data[music_pos1][music_pos2])
    
    string = ' Score metada = ['
    for i in score_meta:
        aux = float(i)
        string += str(aux) + ' '
    string = string.strip()
    string += ']\n\n'
    return string

def precision(sim_data, distances, music_name, title):
    string = ''
    string += "Query = \'" + music_name + ".mp3\'\n\n"
    song_names = np.genfromtxt("Results/song_names.txt", delimiter='\n', dtype=str)

    string += "Ranking: " + title + "-------------\n"
    ranking_d, array_distance = ranking_distance(music_name, song_names, distances)
    string += construct_string(array_distance) + '\n'
    
    string += "\nRanking: Metadata-------------\n"
    ranking_m, array_metadata = ranking_metadata(music_name, song_names, sim_data)
    string += construct_string(array_metadata) + "\n"

    string += score_metadata(song_names, music_name, array_distance, array_metadata, sim_data)

    pre = metric_precision(ranking_d, ranking_m)

    string += "Precision: " + str(pre)
    
    with open("Results/Exercise4/" + music_name + ".mp3/" + title + ".txt", "w") as my_file:
        my_file.write(string)


def metric_precision(ranking_d, ranking_m):
    tp = 0
    for r in ranking_d:
        if r in ranking_m:
            tp += 1
            
    pre = (tp/20) * 100

    return pre


def main():
    warnings.filterwarnings("ignore")

    top100 = "Features/top100_features.csv"

    taffc_metadata = "Dataset/panda_dataset_taffc_metadata.csv"
    
    if(os.path.exists("Results/Exercise2/Exercise_2.1.csv")):
        print("JA EXISTE 2.1")
        top100_normalize = np.genfromtxt("Results/Exercise2/Exercise_2.1.csv", delimiter=',', dtype=float)
    else:
        aux = np.genfromtxt(top100, delimiter=',', dtype=float)[1:,1:]
        aux = aux[:,:aux.shape[1]-1]

        top100_normalize = normalize(aux, "Results/Exercise2/Exercise_2.1.csv")

    if(os.path.exists("Results/Exercise2/Exercise_2.2.csv")):
        print("JA EXISTE 2.2")
        feature_normalize = np.genfromtxt("Results/Exercise2/Exercise_2.2.csv", delimiter=',', dtype=float)
    else:
        feature_normalize = normalize(np.genfromtxt("Expected/FMrosa.csv", delimiter=',', type=float), "Results/Exercise2/Exercise_2.2.csv")
    
    all_array = []

    if(os.path.exists("Results/Exercise3.2/Exercise_3.2_euclidean_top100.csv")):
        print("JA EXISTE 3.2 top 100")
        all_array.append(np.genfromtxt("Results/Exercise3.2/Exercise_3.2_euclidean_top100.csv", delimiter=',', dtype=float))
        all_array.append(np.genfromtxt("Results/Exercise3.2/Exercise_3.2_manhattan_top100.csv", delimiter=',', dtype=float))
        all_array.append(np.genfromtxt("Results/Exercise3.2/Exercise_3.2_cosine_top100.csv", delimiter=',', dtype=float))
    else:
        all_array.append(metrics_euclidean(top100_normalize, "top100"))
        all_array.append(metrics_manhattan(top100_normalize, "top100"))
        all_array.append(metrics_cosine(top100_normalize, "top100"))

    if(os.path.exists("Results/Exercise3.2/Exercise_3.2_euclidean_feature.csv")):
        print("JA EXISTE 3.2 feature")
        all_array.append(np.genfromtxt("Results/Exercise3.2/Exercise_3.2_euclidean_feature.csv", delimiter=',', dtype=float))
        all_array.append(np.genfromtxt("Results/Exercise3.2/Exercise_3.2_manhattan_feature.csv", delimiter=',', dtype=float))
        all_array.append(np.genfromtxt("Results/Exercise3.2/Exercise_3.2_cosine_feature.csv", delimiter=',', dtype=float))
    else:
        all_array.append(metrics_euclidean(feature_normalize, "feature"))
        all_array.append(metrics_manhattan(feature_normalize, "feature"))
        all_array.append(metrics_cosine(feature_normalize, "feature"))

    all_ranks = []
    
    if(os.path.exists("Results/Exercise3.3/Exercise_3.3_manhattan_top100.csv")):
        print("JA EXISTE 3.3 top100")
        all_ranks.append(np.genfromtxt("Results/Exercise3.3/Exercise_3.3_euclidean_top100.csv", delimiter=',', dtype=float))
        all_ranks.append(np.genfromtxt("Results/Exercise3.3/Exercise_3.3_manhattan_top100.csv", delimiter=',', dtype=float))
        all_ranks.append(np.genfromtxt("Results/Exercise3.3/Exercise_3.3_cosine_top100.csv", delimiter=',', dtype=float))
    
    else:
        all_ranks.append(similarity_rankings(all_array[0], "euclidean_top100"))
        all_ranks.append(similarity_rankings(all_array[1], "manhattan_top100"))
        all_ranks.append(similarity_rankings(all_array[2], "cosine_top100")) 

    if(os.path.exists("Results/Exercise3.3/Exercise_3.3_manhattan_feature.csv")):
        print("JA EXISTE 3.3 feature")
        all_ranks.append(np.genfromtxt("Results/Exercise3.3/Exercise_3.3_euclidean_feature.csv", delimiter=',', dtype=float))
        all_ranks.append(np.genfromtxt("Results/Exercise3.3/Exercise_3.3_manhattan_feature.csv", delimiter=',', dtype=float))
        all_ranks.append(np.genfromtxt("Results/Exercise3.3/Exercise_3.3_cosine_feature.csv", delimiter=',', dtype=float))
    
    else:
        all_ranks.append(similarity_rankings(all_array[3], "euclidean_feature"))
        all_ranks.append(similarity_rankings(all_array[4], "manhattan_feature"))
        all_ranks.append(similarity_rankings(all_array[5], "cosine_feature"))
    
    
    if(os.path.exists("Results/Exercise4/Exercise_4.1.2_similarity.csv")):
        print("JA EXISTE 4.1")
        score = np.genfromtxt("Results/Exercise4/Exercise_4.1.2_similarity.csv", delimiter=',', dtype=int)
    else:
        score = song_scores(taffc_metadata)

    names = [ "T100Rosa, Euclidean", "T100Rosa, Manhattan", "T100Rosa, Cosine", "FMRosa, Euclidean", "FMRosa, Manhattan", "FMRosa, Cosine"]

    precision(score, all_array[POS], FNAME, names[POS])

    
if __name__ == "__main__":  
    main()