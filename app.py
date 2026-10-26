"""Grid Controller: local manual gaze-coordinate experiment."""
import json
import math
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from validation import freeze_prediction, label_record
from geometry import intersection, sensitivity, image_point, rotate, cone_overlap

COLORS = ('#00d8ff', '#ffb340')


class App:
    def __init__(self, root):
        self.root = root
        root.title('Grid Controller — Gaze Lab')
        root.geometry('1200x800')
        self.image = None
        self.path = None
        self.rays = [[], []]
        self.scale, self.offset = 1., [0., 0.]
        self.drag = None
        self.ai_queue = queue.Queue()
        self.ai_busy = False
        self.ai_engine = None
        self.ai_data = None
        self.ray_source = "Manual"
        self.mode = tk.StringVar(value='Player 1')
        self.angle = tk.DoubleVar(value=2.)
        self.per_eye = tk.BooleanVar(value=False)
        self.boundaries = tk.BooleanVar(value=True)
        self.info = tk.StringVar(value='Open an original photograph to begin.')
        self.result = tk.StringVar(value='Manual estimate • no gaze model is running')
        self.cursor = tk.StringVar(value='Image coordinates: —')
        self.check = tk.StringVar(value='Click in Inspect mode to record a comparison point.')
        self.clicked = None
        self.validation_snapshot = None
        self.validation_target = None
        bar = ttk.Frame(root, padding=10)
        bar.pack(fill='x')
        ttk.Label(bar, text='GRID CONTROLLER', font=('', 15, 'bold')).pack(side='left', padx=(0, 20))
        for label, command in [('Open image', self.open_image), ('Fit', self.fit),
                               ('Zoom +', lambda: self.zoom(1.25)), ('Zoom −', lambda: self.zoom(.8)),
                               ('Reset lines', self.reset), ('Export JSON', self.export)]:
            ttk.Button(bar, text=label, command=command).pack(side='left', padx=3)
        ai_bar = ttk.Frame(root, padding=(10, 3))
        ai_bar.pack(fill='x')
        self.ai_button = ttk.Button(ai_bar, text='Analyse image (AI)', command=self.analyse_image)
        self.ai_button.pack(side='left', padx=3)
        ttk.Checkbutton(ai_bar, text='Per-eye AI (experimental)', variable=self.per_eye).pack(side='left')
        ttk.Checkbutton(ai_bar, text='Boundary lines', variable=self.boundaries, command=self.draw).pack(side='left')
        self.ai_status = tk.StringVar(value='AI is optional. First run downloads models; photos stay on this computer.')
        ttk.Label(ai_bar, textvariable=self.ai_status, padding=(10, 3), wraplength=600).pack(side='left')
        ttk.Label(root, textvariable=self.info, padding=(10, 3)).pack(anchor='w')
        controls = ttk.Frame(root, padding=10)
        controls.pack(fill='x')
        for mode in ('Player 1', 'Player 2', 'Edit eyes', 'Eye directions', 'Inspect', 'Pan'):
            ttk.Radiobutton(controls, text=mode, variable=self.mode, value=mode, command=self.draw).pack(side='left', padx=8)
        ttk.Label(controls, text='Angle sensitivity ± degrees:').pack(side='left', padx=(20, 5))
        ttk.Spinbox(controls, from_=0, to=15, increment=.5, width=5,
                    textvariable=self.angle, command=self.draw).pack(side='left')
        ttk.Button(controls, text='Apply', command=self.draw).pack(side='left', padx=5)
        ttk.Label(root, text='For each player: click the eyes, then click the estimated gaze direction. '
                  'Drag either handle to refine. Scroll to zoom; right-drag to pan.', padding=10).pack(anchor='w')
        self.canvas = tk.Canvas(root, bg='#111827', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True, padx=10)
        self.canvas.bind('<Button-1>', self.down)
        self.canvas.bind('<B1-Motion>', self.move)
        self.canvas.bind('<ButtonRelease-1>', lambda e: setattr(self, 'drag', None))
        self.canvas.bind('<Button-3>', self.pan_start)
        self.canvas.bind('<B3-Motion>', self.move)
        self.canvas.bind('<ButtonRelease-3>', lambda e: setattr(self, 'drag', None))
        self.canvas.bind('<Motion>', self.hover)
        self.canvas.bind('<MouseWheel>', lambda e: self.zoom(1.2 if e.delta > 0 else 1/1.2, e.x, e.y))
        self.canvas.bind('<Button-4>', lambda e: self.zoom(1.2, e.x, e.y))
        self.canvas.bind('<Button-5>', lambda e: self.zoom(1/1.2, e.x, e.y))
        self.canvas.bind('<Configure>', lambda e: self.draw())
        ttk.Label(root, textvariable=self.cursor, padding=(10, 4)).pack(anchor='w')
        ttk.Label(root, textvariable=self.result, padding=(10, 4), wraplength=1150).pack(anchor='w')
        validation_bar = ttk.Frame(root, padding=(10, 4))
        validation_bar.pack(fill='x')
        ttk.Button(validation_bar, text='Freeze prediction', command=self.freeze_validation).pack(side='left')
        ttk.Radiobutton(validation_bar, text='Mark known ball', variable=self.mode, value='Mark ball').pack(side='left', padx=6)
        self.ball_verified = tk.BooleanVar(value=False)
        ttk.Checkbutton(validation_bar, text='Ball is visible / centre verified', variable=self.ball_verified).pack(side='left')
        ttk.Label(validation_bar, text='Match/group:').pack(side='left', padx=(6, 0))
        self.validation_group = tk.StringVar()
        ttk.Entry(validation_bar, textvariable=self.validation_group, width=14).pack(side='left')
        ttk.Button(validation_bar, text='Save validation JSON', command=self.save_validation).pack(side='left', padx=6)
        self.validation_status = tk.StringVar(value='Validation: freeze a prediction before marking a visible ball. No training runs automatically.')
        ttk.Label(root, textvariable=self.validation_status, padding=(10, 3), wraplength=1150).pack(anchor='w')
        panel = ttk.Frame(root, padding=10)
        panel.pack(fill='x')
        ttk.Label(panel, text='Game comparison: X').pack(side='left')
        self.gx, self.gy = tk.StringVar(), tk.StringVar()
        ttk.Entry(panel, textvariable=self.gx, width=9).pack(side='left', padx=5)
        ttk.Label(panel, text='Y').pack(side='left')
        ttk.Entry(panel, textvariable=self.gy, width=9).pack(side='left', padx=5)
        ttk.Button(panel, text='Compare selected point', command=self.compare).pack(side='left')
        ttk.Label(root, textvariable=self.check, padding=(10, 0, 10, 10)).pack(anchor='w')

    def analyse_image(self):
        if self.image is None:
            return messagebox.showinfo('Open image', 'Open a photograph first.')
        if self.ai_busy:
            return
        self.ai_busy = True
        self.ai_button.configure(state='disabled')
        self.ai_status.set('Loading AI / analysing… First run needs internet to download model weights.')
        source_image = self.image
        snapshot = source_image.copy()
        original_rays = json.dumps([self.rays, self.ai_data])
        use_per_eye = self.per_eye.get()
        def work():
            try:
                from auto_gaze import AutoGaze
                if self.ai_engine is None:
                    self.ai_engine = AutoGaze()
                self.ai_queue.put((source_image, original_rays, self.ai_engine.analyse(snapshot, per_eye=use_per_eye), None))
            except Exception as exc:
                self.ai_queue.put((source_image, original_rays, None, str(exc)))
        threading.Thread(target=work, daemon=True).start()
        self.root.after(150, self.poll_ai)

    def poll_ai(self):
        try:
            source_image, original_rays, data, error = self.ai_queue.get_nowait()
        except queue.Empty:
            self.root.after(150, self.poll_ai)
            return
        self.ai_busy = False
        self.ai_button.configure(state='normal')
        if source_image is not self.image or original_rays != json.dumps([self.rays, self.ai_data]):
            self.ai_status.set('Image or lines changed during analysis. Results discarded; analyse again when ready.')
            return
        if error:
            self.ai_status.set('AI failed; your existing lines are preserved.')
            return messagebox.showerror('AI analysis unavailable', error)
        if not data['faces']:
            self.ai_status.set('No usable faces/eye landmarks detected. Try an original-resolution image or manual lines.')
            return
        self.ai_data = data
        self.ray_source = 'Per-eye AI suggestion' if data.get('per_eye_enabled') else 'AI suggestion'
        self.rays = [face['ray'] or [] for face in data['faces']]
        self.rays += [[] for _ in range(2-len(self.rays))]
        self.angle.set(10.)
        self.ai_status.set(f"Found {data['detected_count']} usable faces; using {len(data['faces'])}, "
                           'largest first, P1 on the left. Review the boxes and lines. '
                           'Missing lines mean the projected direction is unstable. ±10° is a sensitivity setting, not confidence.')
        refined = sum(s == 'iris refined' for f in data['faces'] for s in f.get('eye_status', []))
        total = 2*len(data['faces'])
        self.ai_status.set(self.ai_status.get() + f' Iris refinement: {refined}/{total}; yellow rings are coarse fallback. Use Edit eyes to correct rings.')
        if data.get('per_eye_enabled'):
            self.ai_status.set('Per-eye model: assumed camera calibration. Drag endpoint circles in Eye directions mode. '
                               + ' '.join(f"P{i+1}: {f.get('per_eye_note', '')}" for i, f in enumerate(data['faces'])))
        self.draw()

    def point(self, e):
        return image_point(e.x, e.y, self.scale, self.offset)

    def screen(self, p):
        return (p[0]*self.scale+self.offset[0], p[1]*self.scale+self.offset[1])

    def inside(self, p):
        return self.image and 0 <= p[0] < self.image.width and 0 <= p[1] < self.image.height

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[('Images', '*.jpg *.jpeg *.png *.webp *.bmp'), ('All files', '*')])
        if not path:
            return
        try:
            with Image.open(path) as im:
                im.load()
                self.image = im.convert('RGB')
        except Exception as exc:
            messagebox.showerror('Cannot open image', str(exc))
            return
        self.path, self.rays, self.clicked = path, [[], []], None
        self.clear_validation()
        self.ai_data = None
        self.ray_source = "Manual"
        self.check.set('Click in Inspect mode to record a comparison point.')
        self.info.set(f'{self.image.width:,} × {self.image.height:,} original pixels • '
                      'origin: top left • X right, Y down • no automatic rotation')
        self.fit()

    def fit(self):
        if self.image:
            w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
            self.scale = min(w/self.image.width, h/self.image.height)*.95
            self.offset = [(w-self.image.width*self.scale)/2, (h-self.image.height*self.scale)/2]
            self.draw()

    def zoom(self, factor, x=None, y=None):
        if not self.image:
            return
        x = self.canvas.winfo_width()/2 if x is None else x
        y = self.canvas.winfo_height()/2 if y is None else y
        p = image_point(x, y, self.scale, self.offset)
        self.scale = max(.005, min(16., self.scale*factor))
        self.offset = [x-p[0]*self.scale, y-p[1]*self.scale]
        self.draw()

    def down(self, e):
        if not self.image:
            return
        if self.mode.get() == 'Pan':
            return self.pan_start(e)
        p = self.point(e)
        if not self.inside(p):
            return
        if self.mode.get() == 'Mark ball':
            if self.validation_snapshot is None:
                self.validation_status.set('Freeze the prediction first, then mark the known ball centre.')
                return
            self.validation_target = list(p)
            self.ball_verified.set(False)
            self.update_validation_status()
            self.draw()
            return
        if self.mode.get() == 'Eye directions':
            handles = [(math.dist(self.screen(ray[1]), (e.x, e.y)), i, j)
                       for i, face in enumerate((self.ai_data or {}).get('faces', []))
                       for j, ray in enumerate(face.get('eye_rays', [])) if ray]
            if handles:
                distance, i, j = min(handles)
                if distance < 14:
                    self.drag = ('eye_direction', i, j)
            return
        if self.mode.get() == 'Edit eyes':
            if self.ai_data:
                handles = [(math.dist(self.screen(eye), (e.x, e.y)), i, j)
                           for i, face in enumerate(self.ai_data['faces'])
                           for j, eye in enumerate(face['eyes'])]
                if handles:
                    distance, i, j = min(handles)
                    if distance < 14:
                        self.drag = ('eye', i, j)
                        return
            self.check.set('Zoom into a face and drag a small eye ring. Run AI analysis first.')
            return
        if self.mode.get() == 'Inspect':
            self.clicked = p
            self.check.set(f'Selected original-image point: X={p[0]:.1f}, Y={p[1]:.1f}')
        else:
            if (self.ai_data or {}).get('per_eye_enabled'):
                self.check.set('Use Eye directions to edit per-eye rays, or uncheck Per-eye AI and analyse again.')
                return
            self.ray_source = 'AI + manual edits' if self.ai_data else 'Manual'
            i = int(self.mode.get()[-1])-1
            for j, handle in enumerate(self.rays[i]):
                hx, hy = self.screen(handle)
                if math.hypot(hx-e.x, hy-e.y) < 12:
                    self.drag = ('handle', i, j)
                    return
            if len(self.rays[i]) == 2:
                self.rays[i] = []
            self.rays[i].append(p)
        self.draw()

    def pan_start(self, e):
        self.drag = ('pan', e.x, e.y, *self.offset)

    def move(self, e):
        if not self.drag:
            return
        if self.drag[0] == 'pan':
            _, x, y, ox, oy = self.drag
            self.offset = [ox+e.x-x, oy+e.y-y]
        elif self.drag[0] == 'eye_direction' and self.inside(self.point(e)):
            _, i, j = self.drag
            face = self.ai_data['faces'][i]
            face['eye_rays'][j][1] = list(self.point(e))
            face['per_eye_note'] = 'Direction manually edited; no longer raw AI output.'
            self.ray_source = 'AI + manual eye direction'
        elif self.drag[0] == 'eye' and self.inside(self.point(e)):
            _, i, j = self.drag
            face = self.ai_data['faces'][i]
            old = [sum(eye[k] for eye in face['eyes'])/2 for k in (0, 1)]
            old_eye = face['eyes'][j][:]
            face['eyes'][j] = list(self.point(e))
            eye_ray = face.get('eye_rays', [None, None])[j]
            if eye_ray:
                face['eye_rays'][j] = [[p[k]+face['eyes'][j][k]-old_eye[k] for k in (0, 1)] for p in eye_ray]
                face['per_eye_note'] = 'Eye origin manually moved; direction preserved.'
            face['eye_status'][j] = 'manually corrected'
            new = [sum(eye[k] for eye in face['eyes'])/2 for k in (0, 1)]
            if len(self.rays[i]) == 2:
                # Shift origin and endpoint equally; preserve the estimated direction.
                self.rays[i] = [[p[k]+new[k]-old[k] for k in (0, 1)] for p in self.rays[i]]
            self.ray_source = 'AI + manual eye correction'
        elif self.inside(self.point(e)):
            _, i, j = self.drag
            self.ray_source = 'AI + manual edits' if self.ai_data else 'Manual'
            self.rays[i][j] = self.point(e)
        self.draw()

    def hover(self, e):
        p = self.point(e)
        self.cursor.set(f'Image coordinates: X={p[0]:.1f}, Y={p[1]:.1f}' if self.inside(p) else 'Outside image')

    def reset(self):
        self.clear_validation()
        self.rays = [[], []]
        self.ai_data = None
        self.ray_source = "Manual"
        self.draw()

    def estimate(self):
        return intersection(*self.rays) if all(len(r) == 2 for r in self.rays) else None

    def degrees(self):
        try:
            value = float(self.angle.get())
            return max(0., min(15., value)) if math.isfinite(value) else 2.
        except (ValueError, tk.TclError):
            return 2.

    def eye_overlay(self):
        faces = (self.ai_data or {}).get('faces', [])
        rays = [ray for face in faces for ray in face.get('eye_rays', [])]
        polygon = cone_overlap(rays, self.degrees(), self.image.width, self.image.height) if len(rays) == 4 else []
        return rays, polygon

    def draw_eye_overlay(self):
        rays, polygon = self.eye_overlay()
        if polygon:
            self.canvas.create_polygon(*[v for p in polygon for v in self.screen(p)],
                                       fill='#a3e635', stipple='gray25', outline='#a3e635', width=1)
        extent = math.hypot(self.image.width, self.image.height)*2
        for i, face in enumerate(self.ai_data['faces']):
            for j, ray in enumerate(face.get('eye_rays', [])):
                if not ray or math.dist(*ray) < 1e-8:
                    continue
                variants = [(ray, False)]
                if self.boundaries.get():
                    variants += [(rotate(ray, d), True) for d in (-self.degrees(), self.degrees())]
                for (a, b), boundary in variants:
                    length = math.dist(a, b)
                    end = [a[k]+(b[k]-a[k])*extent/length for k in (0, 1)]
                    self.canvas.create_line(*self.screen(a), *self.screen(end), fill=COLORS[i],
                                            width=1, dash=(3, 4) if boundary else ())
                x, y = self.screen(ray[1])
                self.canvas.create_oval(x-4, y-4, x+4, y+4, outline=COLORS[i])
                self.canvas.create_text(x+7, y, text=f'P{i+1} eye {j+1}', fill=COLORS[i], anchor='w')
        count = sum(bool(r) and math.dist(*r) >= 1e-8 for r in rays)
        status = 'Shared region shaded green.' if polygon else 'No shared region inside the image.'
        if count != 4:
            status = 'Four valid eye directions required for shared-region analysis.'
        self.result.set(f'{self.ray_source}: {count}/4 eye directions. {status} ±{self.degrees():g}° is adjustable '
                        'image-plane tolerance, not confidence. Assumed camera; unvalidated ball location.')

    def draw(self):
        self.canvas.delete('all')
        if not self.image:
            return
        w, h = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        # Render only the viewport: high zoom never creates a giant resized bitmap.
        ox, oy = self.offset
        visible = self.image.transform((w, h), Image.Transform.AFFINE,
            (1/self.scale, 0, -ox/self.scale, 0, 1/self.scale, -oy/self.scale),
            resample=Image.Resampling.BILINEAR, fillcolor='#111827')
        self.photo = ImageTk.PhotoImage(visible)
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        if self.ai_data:
            for i, face in enumerate(self.ai_data['faces']):
                box = face['box']
                self.canvas.create_rectangle(*self.screen(box[:2]), *self.screen(box[2:]),
                                             outline=COLORS[i], dash=(4, 3))
                for j, eye in enumerate(face['eyes']):
                    x, y = self.screen(eye)
                    status = face.get('eye_status', ['coarse fallback']*2)[j]
                    color = '#fde047' if status == 'coarse fallback' else 'white'
                    self.canvas.create_oval(x-4, y-4, x+4, y+4, outline=color, width=1)
                    if self.mode.get() == 'Edit eyes':
                        self.canvas.create_text(x, y-10, text=f'P{i+1} eye {j+1}', fill=color, anchor='s')
        for i, ray in enumerate([] if (self.ai_data or {}).get('per_eye_enabled') else self.rays):
            for j, p in enumerate(ray):
                x, y = self.screen(p)
                self.canvas.create_oval(x-5, y-5, x+5, y+5, fill=COLORS[i], outline='white')
                self.canvas.create_text(x+9, y-12, text=f'P{i+1} '+('eyes' if j == 0 else 'direction'), fill=COLORS[i], anchor='w')
            if len(ray) == 2:
                a, b = ray
                dx, dy = b[0]-a[0], b[1]-a[1]
                length = math.hypot(dx, dy)
                if length:
                    extent = math.hypot(self.image.width, self.image.height)*2
                    end = (a[0]+dx/length*extent, a[1]+dy/length*extent)
                    self.canvas.create_line(*self.screen(a), *self.screen(end), fill=COLORS[i], width=2)
        p = self.estimate()
        self.result.set('Mark two eyes/direction pairs. Results are manual estimates.')
        if not (self.ai_data or {}).get('per_eye_enabled') and all(len(r) == 2 for r in self.rays):
            if p is None:
                self.result.set('No stable forward intersection: rays are parallel, nearly parallel, degenerate, or point apart.')
            else:
                samples = sensitivity(*self.rays, self.degrees())
                for sample in samples:
                    x, y = self.screen(sample)
                    self.canvas.create_oval(x-3, y-3, x+3, y+3, fill='#e879f9', outline='')
                x, y = self.screen(p)
                self.canvas.create_line(x-12, y, x+12, y, fill='white', width=2)
                self.canvas.create_line(x, y-12, x, y+12, fill='white', width=2)
                xs, ys = [q[0] for q in samples], [q[1] for q in samples]
                self.result.set(f'{self.ray_source} intersection: X={p[0]:.1f}, Y={p[1]:.1f} '
                    f'({"inside" if self.inside(p) else "OUTSIDE"} image). '
                    f'±{self.degrees():g}° sampled range: X {min(xs):.0f}–{max(xs):.0f}, '
                    f'Y {min(ys):.0f}–{max(ys):.0f}. Sensitivity only; not confidence or a verified ball location.')
        if (self.ai_data or {}).get('per_eye_enabled'):
            self.draw_eye_overlay()
        if self.validation_target:
            x, y = self.screen(self.validation_target)
            self.canvas.create_oval(x-7, y-7, x+7, y+7, outline='#ff4fa3', width=2)
            self.canvas.create_text(x+10, y+10, text='Known ball (label)', fill='#ff4fa3', anchor='nw')
        if self.clicked:
            x, y = self.screen(self.clicked)
            self.canvas.create_rectangle(x-6, y-6, x+6, y+6, outline='#a3e635', width=2)

    def clear_validation(self):
        self.validation_snapshot = None
        self.validation_target = None
        self.ball_verified.set(False)
        self.validation_status.set('Validation cleared. Freeze a prediction before marking a known ball.')

    def freeze_validation(self):
        if self.image is None:
            return messagebox.showinfo('Open image', 'Open a photograph with a visible ball first.')
        if self.ai_busy:
            return self.validation_status.set('Wait for AI analysis to finish before freezing the prediction.')
        if self.ai_data is None and not all(len(r) == 2 for r in self.rays):
            return self.validation_status.set('Run AI analysis or draw both manual rays first.')
        per_eye = (self.ai_data or {}).get('per_eye_enabled', False)
        self.validation_snapshot = freeze_prediction(
            self.image, self.path.replace('\\', '/').split('/')[-1], self.ray_source,
            self.rays, self.ai_data, None if per_eye else self.estimate(),
            self.eye_overlay()[1] if per_eye else None, self.degrees())
        self.validation_target = None
        self.ball_verified.set(False)
        self.mode.set('Mark ball')
        self.validation_status.set('Prediction frozen. Click the visible ball centre. Later line edits do not change this saved prediction.')
        self.draw()

    def update_validation_status(self):
        record = label_record(self.validation_snapshot, self.validation_target)
        m = record['metrics']
        if m['point_prediction_available']:
            detail = f"Error: {m['error_pixels']:.1f} px ({m['error_percent_diagonal']:.2f}% of image diagonal)."
        elif self.validation_snapshot['prediction']['region'] is not None:
            detail = ('Known ball is inside' if m['target_in_region'] else 'Known ball is outside') + ' the frozen overlap region. No point prediction to score.'
        else:
            detail = 'No forward intersection was predicted; point error is unavailable, not zero.'
        self.validation_status.set('Frozen prediction — ' + detail + ' Verify the label before saving; current lines may differ.')

    def save_validation(self):
        if self.validation_snapshot is None or self.validation_target is None:
            return messagebox.showinfo('Validation', 'Freeze a prediction, then click the known ball centre.')
        if not self.ball_verified.get():
            return messagebox.showinfo('Verify ball label', 'Only mark this verified when the ball is visible and you have checked its centre. A guessed missing-ball position is not ground truth.')
        record = label_record(self.validation_snapshot, self.validation_target,
                              self.validation_group.get(), verified=True)
        path = filedialog.asksaveasfilename(defaultextension='.json', initialfile='validation.json',
                                          filetypes=[('Validation JSON', '*.json')])
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(record, f, indent=2, allow_nan=False)
                self.validation_status.set('Validation saved: frozen prediction, ball label and metrics. No model training performed.')
            except (OSError, ValueError) as exc:
                messagebox.showerror('Validation export failed', str(exc))

    def compare(self):
        if self.clicked is None:
            return self.check.set('Select a point using Inspect first.')
        try:
            x, y = float(self.gx.get()), float(self.gy.get())
            if not all(math.isfinite(v) for v in (x, y)):
                raise ValueError
        except ValueError:
            return self.check.set('Enter finite numeric game X and Y values.')
        dx, dy = x-self.clicked[0], y-self.clicked[1]
        self.check.set(f'Game minus selected point: ΔX={dx:.1f}, ΔY={dy:.1f}; distance={math.hypot(dx, dy):.1f} px. '
                       'Repeat at centre and near all corners; one match is insufficient.')

    def export(self):
        if not self.image:
            return
        path = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('JSON', '*.json')])
        if path:
            data = dict(schema_version=2,
                        per_eye_overlap=self.eye_overlay()[1] if (self.ai_data or {}).get('per_eye_enabled') else None,
                        tolerance_is_confidence=False, image_name=self.path.replace('\\', '/').split('/')[-1],
                        width=self.image.width, height=self.image.height, rays=self.rays,
                        estimated_intersection=None if (self.ai_data or {}).get('per_eye_enabled') else self.estimate(), sensitivity_degrees=self.degrees(),
                        method=self.ray_source + ' gaze rays; unvalidated estimate', ai_analysis=self.ai_data, game_mapping_verified=False)
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            except OSError as exc:
                messagebox.showerror('Export failed', str(exc))


if __name__ == '__main__':
    root = tk.Tk()
    App(root)
    root.mainloop()
