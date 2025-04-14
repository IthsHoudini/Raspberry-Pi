import cv2
import numpy as np

def compute_gradients(image):
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]])

    padded = np.pad(image, ((1,1), (1,1)), mode='edge')
    gx = np.zeros_like(image, dtype=np.float64)
    gy = np.zeros_like(image, dtype=np.float64)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+3, j:j+3]
            gx[i, j] = np.sum(Kx * region)
            gy[i, j] = np.sum(Ky * region)

    return gx, gy

def compute_hog(image, cell_size=8, bin_size=9):
    height, width = image.shape
    gx, gy = compute_gradients(image)
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx) * (180 / np.pi) % 180

    cell_rows = height // cell_size
    cell_cols = width // cell_size
    hog_vector = []

    for i in range(cell_rows):
        for j in range(cell_cols):
            cell_mag = magnitude[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            cell_angle = angle[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]

            histogram = np.zeros(bin_size)
            angle_unit = 180 / bin_size

            for x in range(cell_size):
                for y in range(cell_size):
                    mag = cell_mag[x, y]
                    ang = cell_angle[x, y]
                    bin_idx = int(ang // angle_unit) % bin_size
                    histogram[bin_idx] += mag

            hog_vector.extend(histogram)

    return np.array(hog_vector)

# Start video capture (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))

    hog_desc = compute_hog(gray)

    # Display grayscale image
    cv2.imshow("Live Grayscale Feed", gray)

    # Print only first 10 values of HOG for simplicity
    print("HOG (first 10):", np.round(hog_desc[:10], 2))

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
