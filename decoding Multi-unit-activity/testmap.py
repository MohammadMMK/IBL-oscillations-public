import numpy as np
from pathlib import Path
from brainrender.scene import Scene
from brainrender.video import VideoMaker
from vedo import Points

# Patch the Scene's close method to handle NoneType plotter
def patched_close(self):
    if hasattr(self, 'plotter') and self.plotter is not None:
        self.plotter.close()

Scene.close = patched_close

# Load data
def load_data():
    import pickle
    file_path = 'test_data.pkl'
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    decodability = data['decodability_scores']
    depth_info = data['depth_info']
    x, y, z = depth_info['x_coordinates'], depth_info['y_coordinates'], depth_info['z_coordinates']
    return x, y, z, decodability

# Update scene
def update_scene(scene, x, y, z, decodability, time_index):
    scene.clear_actors()
    current_decodability = decodability[:, time_index]
    coords = np.vstack((x, y, z)).T
    colors = np.interp(current_decodability, (0, 1), (0, 255))
    points = Points(coords, r=10, c=colors)
    scene.add(points)

# Create video
def create_video(x, y, z, decodability, output_path, num_frames):
    scene = Scene(atlas_name=None)  # Disable atlas for testing
    def make_frame(scene, frame_number, nframes=None, resetcam=False, *args, **kwargs):
        update_scene(scene, x, y, z, decodability, frame_number)

    video_maker = VideoMaker(
        scene=scene,
        save_fld=output_path,
        name="decodability_animation",
        fmt="mp4",
        size="1920x1080",
        make_frame_func=make_frame
    )
    video_maker.make_video(duration=num_frames / 10, fps=10)

# Main
if __name__ == "__main__":
    output_folder = Path("output/videos/")
    output_folder.mkdir(parents=True, exist_ok=True)

    x, y, z, decodability = load_data()

    # Testing: Use only a subset
    test_points = 100
    test_timepoints = 10
    x, y, z = x[:test_points], y[:test_points], z[:test_points]
    decodability = decodability[:test_points, :test_timepoints]

    create_video(x, y, z, decodability, output_folder, test_timepoints)
