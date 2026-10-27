"""Grid Controller: local manual gaze-coordinate experiment."""
import json
import math
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
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
        self.scene_engine = None
        self.head_boxes = [None, None]
        self.pending_head = None
        self.scene_overlay_cache = None
        self.scene_view = tk.StringVar(value="Both")
        self.ai_engine = None
        self.ai_data = None
        self.ray_source = "Manual"
        self.mode = tk.StringVar(value='Player 1')
        self.angle = tk.DoubleVar(value=2.)
        self.per_eye = tk.BooleanVar(value=False)
        self.boundaries = tk.BooleanVar(value=True)
        self.result = tk.StringVar(value='Open a photograph to begin.')
        self.cursor = tk.StringVar(value='Image coordinates: —')
        self.clicked = None
        bar = ttk.Frame(root, padding=10)
        bar.pack(fill='x')
        ttk.Label(bar, text='GRID CONTROLLER', font=('', 15, 'bold')).pack(side='left', padx=(0, 20))
        for label, command in [('Open image', self.open_image), ('Fit', self.fit),
                               ('Zoom +', lambda: self.zoom(1.25)), ('Zoom −', lambda: self.zoom(.8)),
                               ('Clear current result', self.reset)]:
            ttk.Button(bar, text=label, command=command).pack(side='left', padx=3)
        self.workflow = 'Scene AI'
        self.workflow_results = {}
        self.mode.set('Inspect')
        self.ai_status = tk.StringVar(value='Open a photo, then choose Analyse scene. You can correct head boxes if needed.')
        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill='x', padx=10, pady=(0, 4))
        scene_bar = ttk.Frame(self.tabs, padding=8)
        ai_bar = ttk.Frame(self.tabs, padding=8)
        manual_bar = ttk.Frame(self.tabs, padding=8)
        for frame, title in ((scene_bar, 'Scene AI'), (ai_bar, 'Eye lines'), (manual_bar, 'Manual')):
            self.tabs.add(frame, text=title)
        self.tabs.bind('<<NotebookTabChanged>>', self.change_workflow)
        self.scene_button = ttk.Button(scene_bar, text='Analyse scene', command=self.analyse_scene)
        self.scene_button.pack(side='left')
        for head in ('Head 1', 'Head 2'):
            ttk.Radiobutton(scene_bar, text='Set '+head.lower(), variable=self.mode, value=head,
                            command=self.update_help).pack(side='left', padx=5)
        ttk.Button(scene_bar, text='Use automatic heads', command=self.clear_heads).pack(side='left', padx=5)
        ttk.Label(scene_bar, text='Show:').pack(side='left', padx=(10, 3))
        view = ttk.Combobox(scene_bar, textvariable=self.scene_view, values=['Both', 'P1', 'P2', 'Hidden'], state='readonly', width=8)
        view.pack(side='left')
        view.bind('<<ComboboxSelected>>', lambda e: self.draw())
        self.ai_button = ttk.Button(ai_bar, text='Analyse eye lines', command=self.analyse_image)
        self.ai_button.pack(side='left')
        ttk.Checkbutton(ai_bar, text='Separate eyes (experimental)', variable=self.per_eye).pack(side='left', padx=8)
        for mode, label in (('Edit eyes', 'Adjust eye positions'), ('Eye directions', 'Adjust directions')):
            ttk.Radiobutton(ai_bar, text=label, variable=self.mode, value=mode,
                            command=self.update_help).pack(side='left', padx=5)
        for mode in ('Player 1', 'Player 2'):
            ttk.Radiobutton(manual_bar, text='Draw '+mode.lower()+' line', variable=self.mode, value=mode,
                            command=self.update_help).pack(side='left', padx=5)
        self.line_settings = ttk.Frame(root, padding=(10, 3))
        ttk.Checkbutton(self.line_settings, text='Show boundary lines', variable=self.boundaries, command=self.draw).pack(side='left')
        ttk.Label(self.line_settings, text='Angle sensitivity ± degrees:').pack(side='left', padx=(15, 5))
        ttk.Spinbox(self.line_settings, from_=0, to=15, increment=.5, width=5,
                    textvariable=self.angle, command=self.draw).pack(side='left')
        ttk.Button(self.line_settings, text='Apply', command=self.draw).pack(side='left', padx=5)
        self.help_text = tk.StringVar(value='Analyse scene estimates looking regions. Optional: set each head box with two corner clicks.')
        self.help_label = ttk.Label(root, textvariable=self.help_text, padding=(10, 3), wraplength=1100)
        self.help_label.pack(anchor='w')
        nav = ttk.Frame(root, padding=(10, 3))
        nav.pack(fill='x')
        for mode, label in (('Inspect', 'Select a point'), ('Pan', 'Move photo')):
            ttk.Radiobutton(nav, text=label, variable=self.mode, value=mode,
                            command=self.update_help).pack(side='left', padx=(0, 10))
        ttk.Label(nav, text='Scroll to zoom · Right-drag to move photo').pack(side='left')
        ttk.Label(root, textvariable=self.ai_status, padding=(10, 3), wraplength=1100).pack(anchor='w')
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

    def change_workflow(self, event=None):
        selected = self.tabs.tab(self.tabs.select(), 'text')
        if selected == self.workflow:
            return
        if self.ai_busy:
            self.tabs.select(('Scene AI', 'Eye lines', 'Manual').index(self.workflow))
            self.ai_status.set('Wait for the current analysis to finish before switching methods.')
            return
        self.workflow_results[self.workflow] = (self.ai_data, self.rays, self.ray_source)
        self.workflow = selected
        self.ai_data, self.rays, self.ray_source = self.workflow_results.get(selected, (None, [[], []], 'Manual'))
        self.pending_head = None
        self.drag = None
        self.scene_overlay_cache = None
        self.mode.set('Player 1' if selected == 'Manual' else 'Inspect')
        self.line_settings.pack_forget()
        if selected != 'Scene AI':
            self.line_settings.pack(fill='x', before=self.help_label)
        self.ai_status.set('Previous result restored.' if self.ai_data or any(self.rays) else 'Ready. '+
                           ('Draw a line for each player.' if selected == 'Manual' else 'Run analysis to get a suggestion.'))
        self.update_help()

    def update_help(self):
        mode = self.mode.get()
        if mode.startswith('Head '):
            text = 'Click two opposite corners around the entire head, including hair. Then choose Analyse scene.'
        elif mode.startswith('Player '):
            text = 'Click the eyes, then a point in the estimated looking direction. Drag either handle to adjust.'
        elif mode == 'Edit eyes':
            text = 'Drag an eye marker to correct its position. This does not retrain the model.'
        elif mode == 'Eye directions':
            text = 'Drag an eye direction handle to adjust the suggested line.'
        elif mode == 'Pan':
            text = 'Drag the photograph to move it. Scroll to zoom.'
        else:
            text = {'Scene AI': 'Analyse scene estimates looking regions. Cyan = player 1; orange = player 2; green = shared region.',
                    'Eye lines': 'Analyse eye lines suggests gaze directions. Adjust markers or directions after analysis.',
                    'Manual': 'Choose a player to draw their estimated looking direction.'}[self.workflow]
        self.help_text.set(text)
        self.draw()

    def analyse_image(self):
        if self.image is None:
            return messagebox.showinfo('Open image', 'Open a photograph first.')
        if self.ai_busy:
            return
        self.ai_busy = True
        self.ai_button.configure(state='disabled')
        self.scene_button.configure(state='disabled')
        self.ai_status.set('Loading AI / analysing… First run needs internet to download model weights.')
        source_image = self.image
        snapshot = source_image.copy()
        original_rays = self.analysis_state()
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
        self.scene_button.configure(state='normal')
        if source_image is not self.image or original_rays != self.analysis_state():
            self.ai_status.set('Image or lines changed during analysis. Results discarded; analyse again when ready.')
            return
        if error:
            self.ai_status.set('AI failed; your existing lines are preserved.')
            return messagebox.showerror('AI analysis unavailable', error)
        if not data['faces']:
            self.ai_status.set('No usable faces/eye landmarks detected. Try an original-resolution image or manual lines.')
            return
        self.ai_data = data
        self.drag = None
        self.scene_overlay_cache = None
        if data.get('scene_enabled'):
            self.ray_source = 'Scene gaze AI'
            self.rays = [[], []]
            self.ai_status.set('Scene model ready. Cyan=P1, orange=P2, green=shared high activation. '
                               'Head boxes are inputs; review them. No action prediction or measured confidence.')
            self.draw()
            return
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

    def analysis_state(self):
        return json.dumps([self.rays, self.ai_data, self.head_boxes, self.pending_head])

    def clear_heads(self):
        self.head_boxes = [None, None]
        self.pending_head = None
        self.mode.set('Inspect')
        self.update_help()
        self.ai_status.set('Automatic head proposals selected. Click Analyse scene.')
        self.draw()

    def analyse_scene(self):
        if self.image is None:
            return messagebox.showinfo('Open image', 'Open a photograph first.')
        if self.ai_busy:
            return
        if self.pending_head is not None:
            return self.ai_status.set('Finish the head box with a second click, or choose Use automatic heads.')
        boxes = [b[:] for b in self.head_boxes if b is not None] or None
        if boxes and self.head_boxes[0] is None:
            return self.ai_status.set('Draw Head 1 first. A single-head analysis uses P1.')
        self.ai_busy = True
        self.ai_button.configure(state='disabled')
        self.scene_button.configure(state='disabled')
        self.ai_status.set('Loading scene model / analysing on CPU. First run downloads a large backbone; this can take several minutes.')
        source = self.image
        snapshot = source.copy()
        state = self.analysis_state()
        def work():
            try:
                from scene_gaze import SceneGaze
                if self.scene_engine is None:
                    self.scene_engine = SceneGaze()
                data = self.scene_engine.analyse(snapshot, boxes=boxes)
                self.ai_queue.put((source, state, data, None))
            except Exception as exc:
                self.ai_queue.put((source, state, None, str(exc)))
        threading.Thread(target=work, daemon=True).start()
        self.root.after(150, self.poll_ai)

    def scene_active(self):
        return bool((self.ai_data or {}).get('scene_enabled'))

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
        self.workflow_results = {}
        self.path, self.rays, self.clicked = path, [[], []], None
        self.head_boxes = [None, None]
        self.pending_head = None
        self.scene_overlay_cache = None
        self.ai_data = None
        self.ray_source = "Manual"
        self.root.title(f'Grid Controller — {self.image.width:,} × {self.image.height:,} pixels')
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
        if self.mode.get() in ('Head 1', 'Head 2'):
            i = int(self.mode.get()[-1])-1
            if self.pending_head is None or self.pending_head[0] != i:
                self.pending_head = (i, p)
                self.ai_status.set('Click the opposite corner of the head box.')
            else:
                a = self.pending_head[1]
                box = [min(a[0], p[0]), min(a[1], p[1]), max(a[0], p[0]), max(a[1], p[1])]
                if box[2]-box[0] < 5 or box[3]-box[1] < 5:
                    return self.ai_status.set('Head box is too small; click a more distant corner.')
                self.head_boxes[i] = box
                self.pending_head = None
                self.ai_status.set('Head box changed. Click Analyse scene to obtain new heatmaps; current results are unchanged.')
            self.draw()
            return
        if self.scene_active() and self.mode.get() != 'Inspect':
            self.ai_status.set('Scene heatmaps cannot be edited as rays. Use Head 1 / Head 2 and analyse again, or switch to Eye lines.')
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
            self.ai_status.set('Zoom into a face and drag a small eye ring. Run AI analysis first.')
            return
        if self.mode.get() == 'Inspect':
            self.clicked = p
            self.ai_status.set(f'Selected original-image point: X={p[0]:.1f}, Y={p[1]:.1f}')
        else:
            if (self.ai_data or {}).get('per_eye_enabled'):
                self.ai_status.set('Use Eye directions to edit per-eye rays, or uncheck Per-eye AI and analyse again.')
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
        self.head_boxes = [None, None]
        self.pending_head = None
        self.scene_overlay_cache = None
        self.drag = None
        self.rays = [[], []]
        self.ai_data = None
        self.ray_source = "Manual"
        self.draw()

    def estimate(self):
        if self.scene_active():
            return None
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
        if self.scene_active() and self.scene_view.get() != 'Hidden':
            from scene_gaze import overlay_image
            key = (id(self.ai_data), self.scene_view.get())
            if self.scene_overlay_cache is None or self.scene_overlay_cache[0] != key:
                self.scene_overlay_cache = (key, overlay_image(self.ai_data['people'], self.scene_view.get()))
            heat = self.scene_overlay_cache[1].transform((w, h), Image.Transform.AFFINE,
                (64/(self.scale*self.image.width), 0, -ox*64/(self.scale*self.image.width),
                 0, 64/(self.scale*self.image.height), -oy*64/(self.scale*self.image.height)),
                resample=Image.Resampling.BILINEAR)
            visible = Image.alpha_composite(visible.convert('RGBA'), heat)
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
        for i, ray in enumerate([] if self.scene_active() or (self.ai_data or {}).get('per_eye_enabled') else self.rays):
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
        if not self.scene_active() and not (self.ai_data or {}).get('per_eye_enabled') and all(len(r) == 2 for r in self.rays):
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
        if self.scene_active():
            people = self.ai_data['people']
            scores = ', '.join(f"P{i+1} in-frame score {p['in_frame_score']:.2f}" for i, p in enumerate(people))
            support = 'Shared high-activation cells shown in green.' if self.ai_data['shared_cell_count'] else 'No shared high-activation region.'
            self.result.set(f'Scene gaze: {scores}. {support} Scores are uncalibrated; 64×64 heatmaps, not exact ball coordinates.')
        for i, box in enumerate(self.head_boxes if self.workflow == 'Scene AI' else []):
            if box:
                self.canvas.create_rectangle(*self.screen(box[:2]), *self.screen(box[2:]), outline=COLORS[i], width=2)
                self.canvas.create_text(*self.screen(box[:2]), text=f'Head {i+1} (next analysis)', fill=COLORS[i], anchor='sw')
        if self.pending_head:
            x, y = self.screen(self.pending_head[1])
            self.canvas.create_oval(x-4, y-4, x+4, y+4, outline='white')
        if self.clicked:
            x, y = self.screen(self.clicked)
            self.canvas.create_rectangle(x-6, y-6, x+6, y+6, outline='#a3e635', width=2)

if __name__ == '__main__':
    root = tk.Tk()
    App(root)
    root.mainloop()
