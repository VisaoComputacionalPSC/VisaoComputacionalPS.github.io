#GRUPO PSC
#OCTAVIO SAVIANO NETO - 11201921665
#TIAGO CORNETTA - 11201922123
#JEFFERSON PAIVA - 11201721192

# EXEMPLO DE CHAMADA: python projeto.py

import cv2
import os
import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import face_recognition
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import threading
from PIL import Image, ImageTk
import os   

dataset_path = "dataset_faces"
path = "/home/ufabc/Documentos/cv2025/projeto"


def detectar_cameras(max_cameras=10):
    print("Procurando por câmeras conectadas...")
    indices_disponiveis = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            print(f"Câmera encontrada no índice {i}")
            indices_disponiveis.append(i)
        cap.release()
    return indices_disponiveis

def tirar_fotos():

    indices = detectar_cameras()
    if len(indices) < 1:
        print("Menos de uma câmera detectadas. Conecte duas e tente novamente.")
        return

    CamL_id = indices[0]

    CamL = cv2.VideoCapture(CamL_id)

    for i in range(100):
        retL, frameL = CamL.read()

    cv2.imshow('Grupo PSC', frameL)
    CamL.release()

    CamL = cv2.VideoCapture(CamL_id)

    start = time.time()
    T = 2
    count = 0

    while True:
        timer = T - int(time.time() - start)
        retL, frameL = CamL.read()

        img1_temp = frameL.copy()
        cv2.putText(img1_temp, f"{timer}", (50, 50), 1, 5, (55, 0, 0), 5)
        cv2.imshow('Grupo PSC', img1_temp)

        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)

        retL, cornersL = cv2.findChessboardCorners(grayL, (8, 6), None)

        if (retL == True) and timer <= 0 and count <20:
            count += 1
            cv2.imwrite(f'{path}/img{count}.png', frameL)

        if timer <= 0:
            start = time.time()

        if (cv2.waitKey(1) & 0xFF == 27) or (count>=20) :
            print("Closing the cameras!")
            break

    CamL.release()
    cv2.destroyAllWindows()

def calibragem():
    print("Extracting image coordinates of respective 3D pattern ....\n")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((8 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

    img_ptsL = []
    img_ptsR = []
    obj_pts = []

    for i in tqdm(range(1, 21)):
        imgL = cv2.imread(f"{path}/img{i}.png")

        imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)

        outputL = imgL.copy()

        retL, cornersL = cv2.findChessboardCorners(outputL, (8, 6), None)

        if retL:
            obj_pts.append(objp)
            cv2.cornerSubPix(imgL_gray, cornersL, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(outputL, (8, 6), cornersL, retL)
            #cv2.imshow('Grupo PSC', outputL)

            img_ptsL.append(cornersL)

            cv2.imwrite(f'{path}/img{i}.png', outputL)

    calcula_params(obj_pts, img_ptsL, imgL_gray)

def calcula_params(obj_pts, img_ptsL, imgL_gray):
    print("Calculating left camera parameters ... ")
    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_pts, img_ptsL, imgL_gray.shape[::-1], None, None)

    Left_Stereo_Map = cv2.initUndistortRectifyMap(
        mtxL, distL, None, mtxL, imgL_gray.shape[::-1], cv2.CV_16SC2
    )

    print("Saving parameters ......")
    cv_file = cv2.FileStorage(f"{path}/params_py.xml", cv2.FILE_STORAGE_WRITE)
    cv_file.write("Left_Stereo_Map_x", Left_Stereo_Map[0])
    cv_file.write("Left_Stereo_Map_y", Left_Stereo_Map[1])
    cv_file.release()

def criar_diretorio_dataset():
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)


def capturar_fotos_rosto():
    num_fotos = 1
    nome_usuario = input("Digite o nome do usuário para capturar as fotos: ").strip()
    if not nome_usuario:
        print("[ERRO] Nome de usuário inválido!")
        return

    cv_file = cv2.FileStorage(f"{path}/params_py.xml", cv2.FILE_STORAGE_READ)
    Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
    Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
    cv_file.release()

    indices = detectar_cameras()
    if len(indices) < 1:
        print("❌ Nenhuma câmera detectada.")
        return

    CamL = cv2.VideoCapture(indices[0])
    criar_diretorio_dataset()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    print(f"[INFO] Capturando {num_fotos} fotos de {nome_usuario}...")

    count = 0
    countdown_time = 3
    start_time = None

    while count < num_fotos:
        retL, frameL = CamL.read()
        if not retL:
            continue

        frameL = cv2.remap(frameL, Left_Stereo_Map_x, Left_Stereo_Map_y, cv2.INTER_LINEAR)
        #grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayL = frameL
        facesL = face_cascade.detectMultiScale(grayL, 1.3, 5)

        rosto = None
        if len(facesL) > 0:
            if start_time is None:
                start_time = time.time()
            elapsed = time.time() - start_time
            countdown = countdown_time - int(elapsed)

            (x, y, w, h) = facesL[0]
            rosto = grayL[y:y+h, x:x+w]
            rosto = cv2.resize(rosto, (200, 200))
            rosto = cv2.GaussianBlur(rosto, (5, 5), 0)  # Suavização de ruído
            cv2.imshow("PSC - Rosto Suavizado", rosto)  # Exibe o rosto suavizado
            cv2.rectangle(frameL, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frameL, f"Contagem: {countdown}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            start_time = None
            countdown = countdown_time

        cv2.putText(frameL, f"Foto {count+1}/{num_fotos}", (10, frameL.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Grupo PSC", frameL)

        if rosto is not None and start_time is not None and elapsed >= countdown_time:
            # Salvar com nome do usuário, e incrementar índice para não substituir
            foto_path = f"{dataset_path}/{nome_usuario}_{count}.png"
            cv2.imwrite(foto_path, rosto)
            print(f"[INFO] Foto {count+1} salva em {foto_path}")
            count += 1
            start_time = None
            time.sleep(0.5)

        if cv2.waitKey(1) & 0xFF == 27:
            print("[INFO] Captura interrompida.")
            break

    CamL.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Captura de fotos para {nome_usuario} concluída.")




def treinar_eigenfaces():
    """
    Carrega as imagens do dataset, treina o modelo Eigenfaces e retorna o reconhecedor treinado.
    """
    print("[INFO] Carregando dataset e treinando Eigenfaces...")

    faces = []
    labels = []
    label_map = {}
    current_label = 0

    path_str = f"{dataset_path}/"

    for filename in os.listdir(path_str):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(path_str, filename)
            # Lê a imagem e converte para grayscale se necessário
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[WARN] Não foi possível carregar a imagem {img_path}")
                continue

            # Extrai o nome do usuário do nome do arquivo
            nome_usuario = filename.rsplit('_', 1)[0]

            if nome_usuario not in label_map:
                label_map[nome_usuario] = current_label
                current_label += 1

            faces.append(img)
            labels.append(label_map[nome_usuario])

    if len(faces) == 0:
        print("[ERRO] Dataset vazio! Capture fotos antes de treinar.")
        return None, None

    recognizer = cv2.face.EigenFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    print("[INFO] Treinamento concluído.")
    return recognizer, label_map



def reconhecer_rosto_eigenface(recognizer, label_map):
    """
    Captura imagem da webcam, aplica calibração com remap, detecta o rosto e tenta reconhecê-lo usando o modelo Eigenfaces.
    """

    if recognizer is None or label_map is None:
        print("[ERRO] Modelo não treinado!")
        return

    # === Ler parâmetros de calibração (remap) ===
    print("[DEBUG] Carregando parâmetros de calibração...")
    cv_file = cv2.FileStorage(f"{path}/params_py.xml", cv2.FILE_STORAGE_READ)
    Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
    Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
    cv_file.release()
    print("[DEBUG] Mapas de calibração carregados com sucesso.")

    # === Inicializar câmera ===
    cam = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    print("[INFO] Posicione seu rosto na frente da câmera para reconhecimento.")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("[ERRO] Falha ao capturar frame da câmera.")
            continue

        # === Aplicar calibração com remap ===
        frame_corrigido = cv2.remap(frame, Left_Stereo_Map_x, Left_Stereo_Map_y, cv2.INTER_LINEAR)

        gray = cv2.cvtColor(frame_corrigido, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        print(f"[DEBUG] {len(faces)} rosto(s) detectado(s).")

        for i, (x, y, w, h) in enumerate(faces):
            rosto = gray[y:y+h, x:x+w]
            rosto = cv2.resize(rosto, (200, 200))

            # Exibe o rosto isolado
            cv2.imshow(f"Grupo PSC {i+1}", rosto)

            # Salva imagem do rosto para inspeção
            debug_img_path = f"debug_rosto_predito_{i+1}.png"
            cv2.imwrite(debug_img_path, rosto)
            print(f"[DEBUG] Rosto {i+1} salvo como {debug_img_path}")

            # === Predição ===
            label_id, confianca = recognizer.predict(rosto)
            print(f"[DEBUG] Rosto {i+1} - Label predito: {label_id}, Confiança: {confianca:.2f}")

            # === Verifica confiança ===
            limiar_confianca = 3500  # Ajuste idealmente com validação
            if confianca > limiar_confianca:
                texto = f"Desconhecido - Confiança: {confianca:.2f}"
                print("[DEBUG] Reconhecimento falhou (desconhecido)")
            else:
                nome_usuario = None
                for nome, idx in label_map.items():
                    if idx == label_id:
                        nome_usuario = nome
                        break
                texto = f"{nome_usuario} - Confiança: {confianca:.2f}"
                print(f"[DEBUG] Reconhecido como {nome_usuario}")

            # === Mostrar na tela ===
            cv2.rectangle(frame_corrigido, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame_corrigido, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # === Mostrar imagem final com anotações ===
        cv2.imshow("Grupo PSC - Reconhecimento Facial (Eigenfaces)", frame_corrigido)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
            print("[INFO] Reconhecimento encerrado pelo usuário.")
            break

    cam.release()
    cv2.destroyAllWindows()

import face_recognition
import cv2
import os
import numpy as np

def reconhecer_rosto_face_recognition():
    dataset_path = "dataset_faces"

    print("[INFO] Carregando imagens do dataset para reconhecimento...")
    imagens = []
    nomes = []
    codificacoes = []

    # Carregar imagens e nomes
    for arquivo in os.listdir(dataset_path):
        if arquivo.endswith(".png") or arquivo.endswith(".jpg"):
            caminho_imagem = os.path.join(dataset_path, arquivo)
            img = face_recognition.load_image_file(caminho_imagem)
            codificacao = face_recognition.face_encodings(img)
            if len(codificacao) > 0:
                codificacoes.append(codificacao[0])
                nome_usuario = arquivo.rsplit('_', 1)[0]
                nomes.append(nome_usuario)
            else:
                print(f"[WARN] Nenhum rosto encontrado na imagem {arquivo}")

    if len(codificacoes) == 0:
        print("[ERRO] Nenhuma codificação facial válida encontrada no dataset!")
        return

    indices = detectar_cameras()
    if len(indices) < 1:
        print("❌ Nenhuma câmera detectada.")
        return

    CamL = cv2.VideoCapture(indices[0])

    print("[INFO] Iniciando reconhecimento. Pressione ESC para sair.")

    while True:
        ret, frame = CamL.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb_frame)
        codificacoes_frame = face_recognition.face_encodings(rgb_frame, faces)

        for (top, right, bottom, left), codificacao_rosto in zip(faces, codificacoes_frame):
            resultados = face_recognition.compare_faces(codificacoes, codificacao_rosto)
            distancias = face_recognition.face_distance(codificacoes, codificacao_rosto)
            melhor_indice = np.argmin(distancias) if len(distancias) > 0 else None

            if melhor_indice is not None and resultados[melhor_indice]:
                nome = nomes[melhor_indice]
                confianca = 1 - distancias[melhor_indice]  # Confiança inversa da distância
            else:
                nome = "Desconhecido"
                confianca = 0

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"{nome} ({confianca:.2f})", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Grupo PSC - Reconhecimento Face_Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
            break

    CamL.release()
    cv2.destroyAllWindows()




def run():
    cal = input("Já tirou foto? (s/n) ")
    if cal == "n":
        tirar_fotos()
        calibragem()
    else:
        calibragem()

    face = input("Já tirou foto face? (s/n) ")
    if face == "n":
        capturar_fotos_rosto()
        recognizer, label_map = treinar_eigenfaces()
    else:
        recognizer, label_map = treinar_eigenfaces()

    metodo = input("Escolha o método de reconhecimento:\n1 - Eigenfaces\n2 - face_recognition\nDigite 1 ou 2: ")

    if metodo == "1":
        reconhecer_rosto_eigenface(recognizer, label_map)
    elif metodo == "2":
        reconhecer_rosto_face_recognition()
    else:
        print("Método inválido. Encerrando.")

class App:
    dataset_path = "dataset_faces"
    def __init__(self, master):
        
        self.master = master
        master.title("Reconhecimento Facial")

        self.menu = tk.Menu(master)
        master.config(menu=self.menu)

        menu_cadastrados = tk.Menu(self.menu, tearoff=0)
        menu_cadastrados.add_command(label="Ver Fotos", command=self.mostrar_cadastrados)
        self.menu.add_cascade(label="Cadastrados", menu=menu_cadastrados)

        self.label = tk.Label(master, text="Projeto Reconhecimento Facial - Grupo PSC")
        self.label.pack(pady=10)

        self.btn_tirar_fotos = tk.Button(master, text="Tirar Fotos para Calibração", command=self.thread_tirar_fotos)
        self.btn_tirar_fotos.pack(pady=5)

        self.btn_calibrar = tk.Button(master, text="Calibrar Câmera", command=self.thread_calibrar)
        self.btn_calibrar.pack(pady=5)

        self.btn_capturar_faces = tk.Button(master, text="Capturar Fotos de Rosto", command=self.thread_capturar_faces)
        self.btn_capturar_faces.pack(pady=5)

        self.btn_treinar = tk.Button(master, text="Treinar Eigenfaces", command=self.thread_treinar)
        self.btn_treinar.pack(pady=5)

        self.label_metodo = tk.Label(master, text="Escolha método de reconhecimento:")
        self.label_metodo.pack(pady=10)

        self.metodo_var = tk.StringVar(value="1")
        self.radio_eigenfaces = tk.Radiobutton(master, text="Eigenfaces", variable=self.metodo_var, value="1")
        self.radio_face_recog = tk.Radiobutton(master, text="face_recognition", variable=self.metodo_var, value="2")
        self.radio_eigenfaces.pack()
        self.radio_face_recog.pack()

        self.btn_reconhecer = tk.Button(master, text="Iniciar Reconhecimento", command=self.thread_reconhecer)
        self.btn_reconhecer.pack(pady=15)

        self.status = tk.Label(master, text="Status: Aguardando ação...", fg="blue")
        self.status.pack(pady=5)

        self.recognizer = None
        self.label_map = None

        self.frame_cadastrados = None

    def mostrar_cadastrados(self):
        janela = tk.Toplevel(self.master)
        janela.title("Fotos Cadastradas")
        janela.geometry("600x500")

        frame = ttk.Frame(janela)
        frame.pack(fill='both', expand=True, padx=10, pady=10)

        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        fotos = []
        colunas = 4
        thumb_size = (120, 120)

        try:
            arquivos = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        except Exception as e:
            arquivos = []
            print(f"Erro ao listar pasta dataset_faces: {e}")

        if not arquivos:
            label = ttk.Label(scrollable_frame, text="Nenhuma foto cadastrada no dataset_faces.")
            label.grid(row=0, column=0)
            return

        for i, arquivo in enumerate(arquivos):
            caminho = os.path.join(dataset_path, arquivo)
            try:
                img = Image.open(caminho)
                img.thumbnail(thumb_size)
                photo = ImageTk.PhotoImage(img)
                fotos.append(photo)  # manter referência para não sumir

                label_img = ttk.Label(scrollable_frame, image=photo)
                label_img.grid(row=(i//colunas)*2, column=i%colunas, padx=10, pady=10)

                # pega o nome do arquivo antes do último underscore
                nome = arquivo.rsplit('_', 1)[0]
                label_nome = ttk.Label(scrollable_frame, text=nome)
                label_nome.grid(row=(i//colunas)*2 + 1, column=i%colunas)
            except Exception as e:
                print(f"Erro ao carregar imagem {arquivo}: {e}")

        janela.fotos = fotos  # mantém referência para não sumir as imagens



    def thread_tirar_fotos(self):
        threading.Thread(target=self.tirar_fotos).start()

    def tirar_fotos(self):
        self.set_status("Iniciando captura de fotos para calibração...")
        tirar_fotos()  
        self.set_status("Captura de fotos finalizada.")

    def thread_calibrar(self):
        threading.Thread(target=self.calibrar).start()

    def calibrar(self):
        self.set_status("Iniciando calibração...")
        calibragem()
        self.set_status("Calibração concluída.")

    def thread_capturar_faces(self):
        threading.Thread(target=self.capturar_faces).start()

    def capturar_faces(self):
        nome_usuario = simpledialog.askstring("Nome do usuário", "Digite o nome do usuário:")
        if not nome_usuario:
            self.set_status("Captura de rosto cancelada: nome não fornecido.")
            return
        self.set_status(f"Iniciando captura de fotos do rosto para {nome_usuario}...")
        capturar_fotos_rosto(nome_usuario, num_fotos=20)  
        self.set_status(f"Fotos do rosto de {nome_usuario} capturadas.")

    def thread_treinar(self):
        threading.Thread(target=self.treinar).start()

    def treinar(self):
        self.set_status("Treinando modelo Eigenfaces...")
        self.recognizer, self.label_map = treinar_eigenfaces()  
        if self.recognizer is not None:
            self.set_status("Treinamento concluído.")
        else:
            self.set_status("Erro no treinamento. Capture fotos antes.")

    def thread_reconhecer(self):
        threading.Thread(target=self.reconhecer).start()

    def reconhecer(self):
        metodo = self.metodo_var.get()
        if metodo == "1":
            if self.recognizer is None or self.label_map is None:
                self.set_status("Modelo não treinado. Treine antes de reconhecer.")
                return
            self.set_status("Iniciando reconhecimento com Eigenfaces...")
            reconhecer_rosto_eigenface(self.recognizer, self.label_map)
            self.set_status("Reconhecimento finalizado.")
        elif metodo == "2":
            self.set_status("Iniciando reconhecimento com face_recognition...")
            reconhecer_rosto_face_recognition()
            self.set_status("Reconhecimento finalizado.")
        else:
            self.set_status("Método inválido selecionado.")

    def set_status(self, mensagem):
        self.status.config(text=f"Status: {mensagem}")

def main():
    root = tk.Tk()
    root.geometry("600x700")
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    run()