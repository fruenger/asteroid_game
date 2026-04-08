from ursina import *
from direct.actor.Actor import Actor
from ursina.shaders import lit_with_shadows_shader
import numpy as np
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
import astropy.units as u
import astropy.time as t
import datetime
from astropy.coordinates import ICRS


######### Generate a new mock asteroid #####
# new coord tuple #
today = datetime.date.today()
obs_time = datetime.time(int(np.random.uniform(21, 4+24) % 24), int(np.random.uniform(0,60)), second=0, tzinfo=datetime.timezone(datetime.timedelta(hours=2)))
obs_location = EarthLocation(
    lon=13.,
    lat=52.,
    height=0.
)
coord = AltAz(
    alt=np.arcsin(np.random.uniform(0.5, 1)) *u.rad, # min. altitude is 30deg
    az=np.random.uniform(0., 2.*np.pi) * u.rad,
    obstime = t.Time(
        datetime.datetime(
            today.year,
            today.month,
            today.day,
            obs_time.hour,
            obs_time.minute,
            obs_time.second,
        ),
    ),
    location=obs_location
)
object_dir_cartesian = np.array([np.cos(coord.alt)*np.cos(coord.az), np.sin(coord.alt), np.cos(coord.alt)*np.sin(coord.az)])

################### define the approximate orbit ##################
import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, solar_system_ephemeris, get_body_barycentric
import astropy.units as u
from poliastro.bodies import Sun
from poliastro.twobody import Orbit

game_paused = False
def pause_game(status):
    game_paused = status

def ra_dec_to_unitvec(ra_deg, dec_deg):
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    return np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec)
    ])

def preliminary_orbit(obs):
    """
    obs = list of three dicts:
        {'ra': deg, 'dec': deg, 'time': 'YYYY-MM-DDTHH:MM:SS', 'location': EarthLocation}
    """

    # 1. Convert RA/Dec to line-of-sight unit vectors
    rho_hat = []
    for ob in obs:
        rho_hat.append(ra_dec_to_unitvec(ob['ra'], ob['dec']))
    rho_hat = np.array(rho_hat)

    # 2. Observer positions in heliocentric frame
    R = []
    with solar_system_ephemeris.set('builtin'):
        for ob in obs:
            t = Time(ob['time'], scale='utc')
            earth = get_body_barycentric('earth', t)
            obsvec = ob['location'].get_gcrs_posvel(t)[0].xyz.to(u.km).value
            # heliocentric observer position (km)
            Rvec = (earth.xyz.to(u.km).value + obsvec)
            R.append(Rvec)
    R = np.array(R)

    # 3. Gauss method (very simplified, no iteration)
    # Use middle observation as reference epoch
    tau1 = (Time(obs[0]['time']).tdb.mjd - Time(obs[1]['time']).tdb.mjd) * 86400.0
    tau3 = (Time(obs[2]['time']).tdb.mjd - Time(obs[1]['time']).tdb.mjd) * 86400.0

    # crude finite-difference derivative of line-of-sight
    drho_dt = (rho_hat[2] - rho_hat[0]) / (tau3 - tau1)

    # assume object distance ~2 AU (asteroid belt) just to close the problem
    r_mag = np.random.uniform(1., 3.) * 1.496e8  # km

    r = R[1] + r_mag * rho_hat[1]
    v = drho_dt * r_mag

    # 4. Make poliastro Orbit object from state vector
    r = r * u.km
    v = v * (u.km/u.s)
    orb = Orbit.from_vectors(Sun, r, v, epoch=Time(obs[1]['time']))

    return orb

# Example use:
loc = EarthLocation.of_site("greenwich")  # any site
coord = coord.transform_to(ICRS())
observations = [
    {'ra': coord.ra.to_value(u.deg), 'dec': coord.dec.to_value(u.deg), 'time': "%i-%i-%iT%i:%i:00" % (today.year, today.month, today.day, obs_time.hour, obs_time.minute), 'location': obs_location},
    {'ra': coord.ra.to_value(u.deg)+0.2, 'dec': coord.dec.to_value(u.deg)+0.02, 'time': "%i-%i-%iT%i:%i:00" % (today.year, today.month, today.day+1, obs_time.hour, obs_time.minute), 'location': obs_location},
    {'ra': coord.ra.to_value(u.deg)+0.4, 'dec': coord.dec.to_value(u.deg)+0.04, 'time': "%i-%i-%iT%i:%i:00" % (today.year, today.month, today.day+2, obs_time.hour, obs_time.minute), 'location': obs_location},
]
orbit = preliminary_orbit(observations)
print(orbit.a, orbit.r_a, orbit.r_p, orbit.ecc, orbit.inc, orbit.period)
#######################################

def day2range(hours):
    """Map the numeric hour to the range 0 to 1."""
    return ((hours + 12) % 24) / 24

def range2day(i):
    """Map the numeric hour to the range 0 to 1."""
    return ((i + 0.5) % 1) * 24

time_since_last_message_trigger = 0
time_now = 0.
min_time = day2range(obs_time.hour - 0.5 + obs_time.minute/60.)
max_time = day2range(obs_time.hour + 0.5 + obs_time.minute/60.)
time_stopped = False

def sun_direction(t: float, latitude: float, sun_declination: float) -> np.ndarray:
    """
    Compute sun direction vector as a unit vector.

    Parameters:
        t (float): Time of day, range [0,1]. 0.5 = noon
        latitude (float): Observer's latitude in degrees
        sun_declination (float): Sun declination in degrees (+N, -S)

    Returns:
        np.ndarray: 3-element unit vector pointing toward the sun
    """
    # Convert degrees to radians
    lat_rad = np.radians(latitude)
    dec_rad = np.radians(sun_declination)

    # Hour angle: 0 at noon, ±π at midnight
    hour_angle = (t - 0.5) * 2.0 * np.pi

    # Altitude (elevation) of the sun
    alt = np.arcsin(-np.sin(dec_rad) * np.sin(lat_rad) +
                    np.cos(dec_rad) * np.cos(lat_rad) * np.cos(hour_angle))

    # Azimuth from south (negative = west, positive = east)
    az = np.arctan2(
        -np.cos(dec_rad) * np.cos(lat_rad) * np.sin(hour_angle),
        -np.sin(dec_rad) - np.sin(lat_rad) * np.sin(alt)
    )

    # Convert spherical to Cartesian coordinates
    x = np.cos(alt) * np.cos(az)
    y = -np.sin(alt)
    z = -np.cos(alt) * np.sin(az)

    # Return normalized unit vector
    return np.array([x, y, z])

def time_str(time_of_day:float):
    """A float value encoding the current time that runs from 0 to 1 with 0.5 being midnight and 0. being noon"""
    hour   = np.floor((time_of_day+0.5)*24) % 24
    minute = np.floor(((time_of_day + 0.5) * 24 * 60) % 60)
    
    return "%s:%s" % (("%i" % hour).rjust(2, "0"), ("%i" % minute).rjust(2, "0"))

def normalize(vec):
    return np.sqrt(np.sum(np.asarray(vec)**2))

class OnScreenMessage(Text):
    
    def __init__(self, message, time_between_letters=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.time_between_letters = time_between_letters
        self.message_triggered = time.time()
        self.message = message
        self.text    = message
        
    def write(self):
        finalliteral = int((time.time() - self.message_triggered) / self.time_between_letters)
        finalliteral = np.clip(finalliteral, 0, len(self.message))
        self.text = self.message[:finalliteral]
        
    def reset_timer(self):
        self.message_triggered = time.time()

def get_dome_intersect(sample_radius, origin, direction):
    """\vec{r}\left(R\right)=\vec{r}_{0}+\vec{a}\sqrt{\left(\vec{r}_{0}\cdot\vec{a}\right)^{2}+\left(R^{2}-r_{0}^{2}\right)}-\left(\vec{r}_{0}\cdot\vec{a}\right)\vec{a}"""
    v_r0 = np.asarray(origin)
    v_a  = np.asarray(direction)
    
    a2 = np.dot(v_a, v_a)
    
    result = v_r0 + v_a * (np.sqrt(np.dot(v_r0, v_a)**2 / a2**2 + sample_radius**2/a2 - np.dot(v_r0, v_r0)/a2) - np.dot(v_r0, v_a)/a2)
    return result

def blink_opacity(duration, min_alpha=0.25):
    return min_alpha + (np.sin(np.pi * time.time() / duration)**2) * (1.-min_alpha)

def load_shader(filename):
    with open(filename, "r") as f:
        result = f.read()
        
    return result

app = Ursina()
viewer = EditorCamera()
window.fullscreen = True
window.fps_counter.enabled = False
window.entity_counter.enabled = False
window.collider_counter.enabled = False
window.borderless = True
camera.fov = 90.





help_texts = {
    0 : "Ein noch unbekanntes\nkleines Objekt wurde in den Tiefen des Sonnensystems mit einem robotischen Teleskop gesichtet. Heute Nacht ist es unsere Aufgabe, die Nachverfolgung dessen aufzunehmen. dazu muessen wir zunachst warten, bis es ueber dem Horizont steht. Heute Nacht ist das zwischen %s und %s Uhr. Warte bis wir im Zeitfenster sind und druecke die Taste [LEERTASTE], um die Zeit zu pausieren." % (time_str(min_time), time_str(max_time)),
    1 : "Nun muss das Observatorium geöffnet und die Technik drinnen hochgefahren werden. Mit dem betaetigen der Z-Taste kann das bewerkstelligt werden.",
    2 : "Sobald das observatorium vollständig geöffnet ist, steuere das Teleskop an die vorgesehen Himmelsposition. Diese ist mit einen kleinen roten Punkt am Himmel markiert. Zum besseren Orientierung benutzen wir heute unseren Navigierlaser (blauer Strahl). Die Moniterung des Teleskops wird mit [LINKS] und [RECHTS] auf der RA-Achse gefahren und mit [OBEN] und [UNTEN] auf der DEC-Achse. Mithilfe beider können jeden beliebigen Ort am Himmel anpeilen. Das ist nun Deine Aufgabe!",
    3 : "Das Teleskop schaut nun auf den Himmelsbereich unserer Wahl. Allerdings muss auch die Kuppel korrekt ausgerichtet sein. Der Kuppelschlitz muss nun so ausgerichtet sein, dass wir freie Sicht auf den Himmel durch unser Teleskop haben. Mit den Pfeiltasten [LINKS] und [RECHTS] kannst du nun auch die Kuppel selbst drehen, sodass wir unser Beobachtungen beginnen können.",
    4 : "Mache *drei* Bilder hinereinander und warte bis das jeweilige Bild ein gutes Rauschverhältnis hat, um man rund 20 Sterne klar erkennen kann. Wenn du zufrieden mit dem Bild bist, dann drücke [LEERTASTE]",
    5 : "Die drei separaten Bilder sind nun als ein Blinkbild dargestellt. In diesem Bild gibt es nun ein Objekt das sich über das STernfeld bewegt - das ist der Asteroid, den wir beobachten. Wähle den sich bewegenden Punkt mit der [LINKEN MAUSTASTE] aus, um dessen Position am Himmel zu bestätigen.",
    6 : "Die Beobachtung war erfolgreich! Das neu entdeckte Objekt benötigt noch einen Namen für den Eintrag in unseren OST Asteroidenkatalog."
}

def wrap_text(text: str, width: int) -> str:
    """Wrap text to given width, breaking only on spaces."""
    words = text.split()
    lines = []
    line = []

    for w in words:
        # length if we add this word to current line
        if sum(len(x) for x in line) + len(line) + len(w) <= width:
            line.append(w)
        else:
            lines.append(" ".join(line))
            line = [w]
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)

class HelpWindow(TextField):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def update(self):
        global step
        if self.enabled:
            self.text = wrap_text(help_texts[step], 80)
            self.render()
            
    def fade_in(self, duration=1., update_frq=20.):
        for alpha_value in np.linspace(0., 1., int(duration * update_frq)):
            invoke(self.bg.alpha_setter, alpha_value, delay=alpha_value * duration)
            invoke(self.text_entity.alpha_setter, alpha_value, delay=alpha_value * duration)
            
    def fade_out(self, duration=1., update_frq=20.):
        for inverse_alpha_value in np.linspace(0., 1., int(duration * update_frq)):
            invoke(self.bg.alpha_setter, 1.- inverse_alpha_value, delay=inverse_alpha_value * duration)
            invoke(self.text_entity.alpha_setter, 1.- inverse_alpha_value, delay=inverse_alpha_value * duration)


help_window = HelpWindow(character_limit=99999, x=0, y=0, origin=(-0.5,-0.5), max_lines=64, line_height=2.)
help_window.position = (-0.4 * camera.aspect_ratio_getter(), 0.4, 0.03)
help_window.text_entity.position -= Vec3(-0.05, 0.05, 0)
help_window.bg.scale = (0.8 * camera.aspect_ratio_getter(), 0.8)
help_window.active = False
help_window_close_button = Button("X", parent=help_window, color=color.red, text_color=color.white, scale=0.05)
help_window_close_button.position = (0.8 * camera.aspect_ratio_getter(), 0)

def close_help_window():
    global game_paused
    help_window.fade_out(0.5)
    invoke(help_window.enabled_setter, False, delay=0.5)
    game_paused = False
        
help_window_close_button.on_click = close_help_window






# the telescope base platform
telescope_base = Entity(
    model="models/telescope_mount_base.glb",
    shader=lit_with_shadows_shader,
    color=color.gray,
    position=Vec3(0,0,0)
)

base_pivot = Entity(
    position=Vec3(-1,3,0),
    rotation=Vec3(0, 0, 52-90),
    parent=telescope_base
)

ra_pivot = Entity(
    #model="cube",
    #color=color.green,
    position=Vec3(0,0,0),
    parent=base_pivot
)
dec_pivot = Entity(
    #model="cube",
    #color=color.red,
    position=Vec3(1.8,1.15,0),
    parent=ra_pivot
)

class Skybox(Entity):
    
    def __init__(self):
        super().__init__(
            model="models/skybox.glb",
            shader=Shader(
                vertex=load_shader("shaders/sky.vert"),
                fragment=load_shader("shaders/sky.frag")
            ),
            unlit=True,
            parent=camera
        )
        self.setBin('background', 0)
        self.scale = camera.clip_plane_far * .8/100 # the blender object has a radius of 100. So we need to tune it down a bit
        
    def update(self):
        self.world_rotation = Vec3(0, 0, 0)

sky = Skybox()
sky.set_shader_input("u_time", 0)
sky.hide(0b0001)

# the ra_axis body
telescope_base2 = Entity(
    model="models/telescope_mount_ra.glb",
    shader=lit_with_shadows_shader,
    color=color.gray,
    position=Vec3(-1,3,0),
    rotation=Vec3(0, 0, 52-90)
)
telescope_ra = Entity(
    model="models/telescope_mount_dec.glb",
    shader=lit_with_shadows_shader,
    color=color.gray,
    parent=ra_pivot
)

telescope_dec = Entity(
    model="models/telescope_tube.glb",
    shader=lit_with_shadows_shader,
    color=color.gray,
    parent=dec_pivot,
    rotation=(0,90,0),
    position=(2.2,-0.5,0)
)

telecope_optical_axis = Entity(
    #model="cube",
    position=Vec3(2.2, 0., 0),
    scale=Vec3(1,10,1),
    color=color.red,
    parent=dec_pivot
)

######### DOME ##########
domewall = Entity(
    model="models/observatory_domewall.glb",
    shader=lit_with_shadows_shader,
    color=color.gray,
    scale=Vec3(15,15,15)
)
dome_pivot = Entity(
    model="models/observatory_dome.glb",
    position=Vec3(0,0,0),
    shader=lit_with_shadows_shader,
    color=color.gray,
    scale=Vec3(15,15,15),
    collider='mesh'
)

shutter_pivot = Entity(
    position=Vec3(0,0,0),
    parent=dome_pivot
)
shutter = Entity(
    model="models/observatory_shutter.glb",
    parent=shutter_pivot,
    shader=lit_with_shadows_shader,
    color=color.gray
)
flap_pivot = Entity(
    position=Vec3(1,0,0),
    parent=dome_pivot,
    shader=lit_with_shadows_shader,
    color=color.gray
)
flap = Entity(
    model="models/observatory_flap.glb",
    position=Vec3(-1,0,0),
    parent=flap_pivot,
    shader=lit_with_shadows_shader,
    color=color.gray
)
bottom = Entity(
    model='plane',
    position=Vec3(0, -7.5, 0),
    scale=Vec3(1000, 1, 1000),
    color=color.gray,
    shader=lit_with_shadows_shader
)
"""screen = Entity(
    model='plane',
    position=Vec3(0, 15, 0),
    scale=Vec3(5,5,5),
    shader=Shader(
        vertex=load_shader("shaders/screen.vert"),
        fragment=load_shader("shaders/screen.frag")
    )
)"""


light_box = Entity(
    model="cube",
    scale=100.,
    color=color.rgb(0,0,0,0)
)
light = DirectionalLight(shadows=True)
light.update_bounds(entity=light_box)

########## Overlay
class TransitionMask(Entity):
    
    def __init__(self, *args, **kwargs):
        super().__init__(model='quad', color=color.black, parent=camera.ui, scale=(camera.aspect_ratio_getter(), 1), *args, **kwargs)
        self.position = Vec3(0,0,0.01)
        self.alpha_setter(0)
        
    def fade_in(self, duration=1., update_frq=20.):
        for alpha_value in np.linspace(0., 1., int(duration * update_frq)):
            invoke(self.alpha_setter, alpha_value, delay=alpha_value * duration)
            
    def fade_out(self, duration=1., update_frq=20.):
        for inverse_alpha_value in np.linspace(0., 1., int(duration * update_frq)):
            invoke(self.alpha_setter, 1.- inverse_alpha_value, delay=inverse_alpha_value * duration)
            self.alpha_setter(1)
            
tr_mask = TransitionMask()


#actor.loop("Action")
#cube_model = Actor()
#cube_model.reparent_to(etty)

telescope_speed_ra = 0.
telescope_speed_dec = 0.
dome_speed_az       = 0.

MAX_SPEED = 10.
ACCEL     = 20.

sun_dec = 0.5 # deg
sky.set_shader_input("sun_dec", sun_dec)

infotext     = OnScreenMessage(message="[default text]", time_between_letters=0.01, origin=(-0.5, 0.5), parent=camera.ui, position=[-.8, -.42,-.002])
prev_message = infotext.message
time_display = Text(text="hello world", origin=(-0.5, 0.5), parent=camera.ui, position=[-0.8, 0.45, 0.015], scale=2)
step         = 0

### EVENT LINE #####. Every entry will be executes EXAXCTLY once upon level-up
def stageup_event(stage):
    """Specify the number of the stage that you are starting in now."""
    global last_update, cheat_through, game_paused
    
    if stage == 2:
        # start the laser beam
        line = Entity(model=Mesh(vertices=[Vec3(0,0,0), Vec3(0,1000,0)],
            mode='line',
            thickness=2),
            color=color.azure,
            parent=telecope_optical_axis
        )
        
        marker = Entity(
            model='sphere',
            color=color.red,
            position=0.6*camera.clip_plane_far*object_dir_cartesian,
            scale=[50, 50, 50]
        )
        
    if stage == 4:
        pause_game(True)
        invoke(pause_game, False, delay=5)
        tr_mask.fade_in(2.5)
        invoke(image_panel.enabled_setter, True, delay=2.5)
        invoke(tr_mask.fade_out, 2.5, delay=2.5)
        last_update = time.time()
        cheat_through = False############
        

#### Create a fake_image ###### (for step 4)
from scipy.ndimage import gaussian_filter
imsize = 250
Nstars = 100

# simulate the trajectory of the object
traj_angle = np.random.uniform(0, 2. * np.pi)
image_locations = np.array([[np.cos(traj_angle), np.sin(traj_angle)], [-np.cos(traj_angle), -np.sin(traj_angle)]])

# put the third location as the average location 
image_locations  = np.insert(image_locations, 1, np.mean(image_locations, axis=1), axis=1)
image_locations *= np.random.uniform(4., 20.)
image_locations += np.expand_dims(np.random.uniform(imsize / 4., imsize * 3./ 4., 2), axis=1)
image_locations  = image_locations.astype(int)

image_perfect_bg = np.ones((imsize, imsize), dtype=float) / 50.
image_perfect_bg[np.random.randint(0, imsize, Nstars), np.random.randint(0, imsize, Nstars)] = np.random.exponential(0.5, Nstars)

all_images_perfect = []
for x, y in image_locations.T:
    image_perfect = image_perfect_bg.copy()
    image_perfect[x, y] = 1.
    image_perfect = gaussian_filter(image_perfect, sigma=1.5)
    all_images_perfect.append(image_perfect)

last_update = 0
exposure_time = 0.
from PIL import Image
all_images = []
image_array = np.zeros((imsize, imsize))
tex = Texture(Image.fromarray(image_array.astype(np.uint8), mode="L").convert("RGBA"))
image_panel = Entity(
    model='quad',
    parent=camera.ui,
    #rotation=[-90,0,0],
    enabled=False,
    collider='box',
    position=(0,0,0.04),
    scale=0.8,
    origin=(0,0)
)

# for step 5
image_shown = 0



def update():
    ### ### ### check for user input and change the telescope position ### ### ###
    global time_now, time_stopped, step, telescope_speed_ra, telescope_speed_dec, MAX_SPEED, dome_speed_az, cheat_through, last_update, image_array, imsize, all_images, exposure_time, image_shown, all_images_perfect, prev_message
    
    # execute the following commands no matter whether the game is paused or not
    if prev_message != infotext.message:
        infotext.reset_timer()
    
    # make the text blink
    infotext.write()
    infotext.wordwrap_setter(100)
    infotext.alpha_setter(blink_opacity(3))
    
    if game_paused:
        return None
    
    if step == 2:
        telescope_speed_ra = np.clip(telescope_speed_ra + ACCEL * time.dt * held_keys["right arrow"] - ACCEL * time.dt * held_keys["left arrow"], -MAX_SPEED, MAX_SPEED)
        telescope_speed_ra -= ACCEL / 2. * time.dt * np.sign(telescope_speed_ra)
        ra_pivot.rotation_y += telescope_speed_ra * time.dt
        
        telescope_speed_dec = np.clip(telescope_speed_dec + ACCEL * time.dt * held_keys["up arrow"] - ACCEL * time.dt * held_keys["down arrow"], -MAX_SPEED, MAX_SPEED)
        telescope_speed_dec -= ACCEL / 2. * time.dt * np.sign(telescope_speed_dec)
        dec_pivot.rotation_x += telescope_speed_dec * time.dt
    
    if step == 3:
        dome_speed_az = np.clip(dome_speed_az + ACCEL * time.dt * held_keys["left arrow"] - ACCEL * time.dt * held_keys["right arrow"], -MAX_SPEED, MAX_SPEED)
        dome_speed_az -= ACCEL / 2. * time.dt * np.sign(dome_speed_az)
        dome_pivot.rotation_y += dome_speed_az * time.dt
    
    #dec_pivot.world_rotation_z # altitude, or dec_pivot.up # is the pointing direction in cartesian space
    
    # let time run forward
    
    if not time_stopped:
        time_now = (time.time() / 1e2) % 1.
        sky.set_shader_input("u_time", time_now)
    
    sun_dir = sun_direction(time_now, 52., sun_dec)
    light.look_at(-sun_dir)
    
    time_display.text = time_str(time_now)
    
    # open the dome at night
    prev_message = infotext.message
    if step == 0:
        infotext.message = "Das Beobachtungsfenster ist zwischen %s und %s Uhr heute Nacht. Warte und druecke [LEERTASTE], um die Beobachtung zu beginnen." % (time_str(min_time), time_str(max_time))
        infotext.color = color.green
        
    if step == 1:
        
        infotext.message = "Öffne das Observatorium mit [Z]."
        infotext.color = color.green
            
    # align the telescope to the respective position in the sky
    if step == 2:

        infotext.message = "Richte das Teleskop mit den Pfeiltasten auf das Ziel aus."
        infotext.wordwrap_setter(100)
        infotext.color = color.green
        
        ########################## CONTINUE HERE ......
        if np.dot(object_dir_cartesian, telecope_optical_axis.up / normalize(telecope_optical_axis.up)) > np.cos(np.radians(0.5)):
            step = 3

    if step == 3 or cheat_through:
        
        infotext.message = "Richte die Kuppel mit [←] und [→] auf das Objekt aus."
        infotext.wordwrap_setter(100)
        infotext.color = color.green
        
        rayinfo = raycast(telecope_optical_axis.world_position, telecope_optical_axis.up, traverse_target=dome_pivot)
        
        if rayinfo.distance > 100. or cheat_through:
            step = 4
            stageup_event(4)
    
    if step == 4:
        
        infotext.message = "Die Kamera belichtet ... Drücke [LEERTASTE], um ein Bild zu machen"
        infotext.wordwrap_setter(100)
        infotext.color = color.green
        exposure_time += time.dt
        
        ############ Create a picture ############
        if time.time() - last_update > 0.1:
            image_perfect = all_images_perfect[len(all_images)]
            image_array += np.random.poisson(image_perfect*1., (imsize,imsize)) #<<<<<<<<<<<<<<<<<<<<<< change speed of exposure here (1 is normal speed)
            image_panel.texture = Texture(Image.fromarray(((image_array / image_array.max())*255).astype(np.uint8), mode="L").convert("RGBA"))
            last_update = time.time()
        
        if held_keys["space"]:
            if exposure_time >= 3.:
                all_images.append(image_array.copy())
                image_array = np.zeros((imsize, imsize))
                last_update = time.time()
                exposure_time = 0.
        
            if len(all_images) == 3:
                step = 5
                last_update = time.time()
                #tr_mask.fade_in(1.)
                #invoke(tr_mask.fade_out, 1., delay=1.)
                
    if step == 5:
        
        infotext.message = "Wähle das sich bewegende Objekt mit der [linken Maustaste] aus. (Mit [R] kannst du auf die drei Aufnahmen noch einmal von vorn starten falls du auf deinen Bildern das Objekt noch nicht erkennen kannst.)"
        infotext.wordwrap_setter(100)
        
        if time.time() - last_update > 0.5:
            image_panel.texture = Texture(Image.fromarray(((all_images[image_shown] / all_images[image_shown].max())*255).astype(np.uint8), mode="L").convert("RGBA"))
            image_shown += 1
            image_shown = int(image_shown % 3)
            last_update = time.time()
        
        # redo the imaging series if wanted
        if held_keys["r"]:
            step = 4
            all_images = []
            exposure_time = 0.
            image_array = np.zeros((imsize, imsize))
            last_update = time.time()
        
    if step == 6:
        pass

cheat_through = False ################## CHEAT

sun_dir = [0,0,1]
update()

def input(key):
    global sun_dir, step, time_now, time_stopped, imsize, tr_mask
    
    # this combination let's you advance to the next level
    if key == "space" and step == 0:
        if min_time < time_now and max_time > time_now:
            time_stopped = True
            step = 1
            
        else:
            print_on_screen("Noch nicht der richtige Zeitpunkt zum Beobachten des Objekts ...", origin=(0,0), color=color.red, duration=2)
            
    
    if key == "z" and step == 1:
        step = 2
        shutter_pivot.animate_rotation([0.,0.,-67.5], duration=10., curve=curve.linear)
        flap_pivot.animate_rotation([0.,0.,80.], duration=10., curve=curve.linear)
        invoke(stageup_event, 2, delay=10) # make the laser beam appear shortly after the dome has been opened
        
    if key == "x":
        shutter_pivot.animate_rotation([0.,0.,0.], duration=10., curve=curve.linear)
        flap_pivot.animate_rotation([0.,0.,0.], duration=10., curve=curve.linear)
    
    if key == "h":
        help_window.enabled = True
        help_window.fade_in(0.5)
        game_paused = True
    
    if key == "i":
        
        dome_intersect = get_dome_intersect(
            np.sqrt(np.sum(flap_pivot.world_position**2)),
            telecope_optical_axis.world_position,
            telecope_optical_axis.up
        )
        target_azimuth = np.degrees(np.arctan2(dome_intersect[2], dome_intersect[0]))
        dome_pivot.animate_rotation([0., -target_azimuth, 0.], duration=2, curve=curve.linear)

    if key == 'left mouse down' and step == 5:
        
        world_point = mouse.world_point      # 3D world position on the quad
        try:
            local_point = image_panel.get_relative_point(scene, world_point)  # convert to panel's local coords
            local_point = (0.5 + np.array([-local_point[1], local_point[0]])) * imsize
            distance = np.min(np.sqrt(np.sum((np.expand_dims(local_point, axis=1) - image_locations)**2, axis=0)))
            print("[DEBUG] Distance to object: %i.1fpx" % distance)
            
            if distance / imsize < 0.05:
                step = 6
                tr_mask.fade_in(1.)
                invoke(tr_mask.fade_out, 1., delay=1.)
                invoke(image_panel.enabled_setter, False, delay=1.)
                
                # level up to step 6
                step6()
                
        except TypeError as te:
            print("error", te)

def step6():
    infotext.enabled = False
    
    infotext.message = ""
    
    obj_name_label = Text("Objektname", x=-0.4 * camera.aspect_ratio_getter(), y=0.4, origin=(-0.5, 0.5))
    obj_name = TextField(character_limit=30, max_lines=1, x=-0.4 * camera.aspect_ratio_getter(), y=0.36, origin=(-0.5, 0.5))
    obj_name.bg.scale_x = 0.5
    obj_name.bg.scale_y = 0.05
    obj_name.add_text("OST/%s" % datetime.datetime.now().strftime(r"%Y-%m-%d %H-%M-%S"))
    obj_name.active = False
    
    discoverer_name_label = Text("Endeckerteam", x=-0.4 * camera.aspect_ratio_getter(), y=0.25, origin=(-0.5, 0.5))
    discoverer_name = TextField(character_limit=30, max_lines=1, x=-0.4 * camera.aspect_ratio_getter(), y=0.21, origin=(-0.5, 0.5))
    discoverer_name.text_entity.color = color.green
    discoverer_name.bg.scale_x = 0.5
    discoverer_name.bg.scale_y = 0.05
    discoverer_name.add_text("Name d. Entdeckerteams")
    discoverer_name.active = True
    
    orbital_elements_label = Text("Bahnelemente", x=-0.4 * camera.aspect_ratio_getter(), y=0.05, origin=(-0.5, 0.5))
    orbital_elements = TextField(character_limit=30, max_lines=8, x=-0.4 * camera.aspect_ratio_getter(), y=0.0, origin=(-0.5, 0.5))
    orbital_elements.bg.scale_x = 0.5
    orbital_elements.bg.scale_y = 0.25
    orbital_elements.add_text("""a = %.2f AE
e = %.3f 
i = %.2f°
Ω = %.1f°
ω = %.1f°
P = %.2f Jahre
Pe = %.2f AE
Ap = %.2f AE
    """ % (
        orbit.a.to_value(u.AU),
        orbit.ecc,
        orbit.inc.to_value(u.deg),
        orbit.raan.to_value(u.deg),
        orbit.argp.to_value(u.deg),
        orbit.period.to_value(u.yr),
        orbit.r_a.to_value(u.AU),
        orbit.r_p.to_value(u.AU)
    ))
    orbital_elements.active = False
    
    img_view = Entity(model='quad', parent=camera.ui, x=0.7, y=0., scale=(0.8, 0.8), origin=(0.5, 0.))
    coadded_image = np.sum(all_images, axis=0)
    img_view.texture = Texture(Image.fromarray(((coadded_image / coadded_image.max())*255).astype(np.uint8), mode="L").convert("RGBA"))
    save_button = Button("Add to Catalog", highlight_scale=1.1, pressed_scale=0.95, scale=(0.25, 0.1), x=-0.4 * camera.aspect_ratio_getter(), y=-0.35, origin=(-0.5, 0.5))
    
    def save_button_fct():
        with open("user_saves.dat", "a") as f:
            
            f.write("%s, %s, %s\n" % (datetime.datetime.now().strftime(r"%Y-%m-%d %H:%M:%S"), obj_name.text, discoverer_name.text))
            python = sys.executable # restart the app
            os.execl(python, python, *sys.argv)
            
    save_button.on_click = save_button_fct
    #sosy_button = Button("Solar System Viewer", highlight_scale=1.1, pressed_scale=0.95, scale=(0.25, 0.1), x=-0.25 * camera.aspect_ratio_getter(), y=-0.35, origin=(-0.5, 0.5))
    
app.run()