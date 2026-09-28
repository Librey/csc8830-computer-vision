"""
CSc 8830 Computer Vision, Module 4, Question 3: figures for the theory write-up

README
------
Install:  pip install numpy opencv-python matplotlib
Run:      python q3_make_figures.py
Writes fig_edges.pdf, fig_ringing.pdf, fig_filters.pdf and fig_texture.pdf,
which q3_fourier_edges_segmentation.tex includes. Every filter here runs in
the frequency domain: FFT, multiply by H(u,v), inverse FFT.
Expects the Q1 astronaut image at ../q1/images/astronaut.png.
"""
import numpy as np, cv2, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

def freq_grid(M,N):
    u=np.fft.fftfreq(M)[:,None]; v=np.fft.fftfreq(N)[None,:]   # cycles/pixel
    return u,v,np.sqrt(u**2+v**2)

def apply(f,H): return np.real(np.fft.ifft2(np.fft.fft2(f)*H))

# ---------- Figure 1: edge detection in the frequency domain ----------
f=cv2.imread('../q1/images/astronaut.png',0).astype(float)/255
M,N=f.shape; u,v,D=freq_grid(M,N)
D0=0.05
H_ideal=(D>D0).astype(float)
H_gauss=1-np.exp(-D**2/(2*D0**2))
sigma=2.0
H_log=-4*np.pi**2*D**2*np.exp(-2*np.pi**2*sigma**2*D**2)
g_ideal=apply(f,H_ideal); g_gauss=apply(f,H_gauss); g_log=apply(f,H_log)
# zero crossings with slope threshold
zc=np.zeros_like(g_log,bool)
for a,b in [(g_log[:,:-1],g_log[:,1:]),(g_log[:-1,:],g_log[1:,:])]:
    c=(np.sign(a)!=np.sign(b))&(np.abs(a-b)>1.0*np.std(g_log))
    if a.shape[1]<N: zc[:,:-1]|=c
    else: zc[:-1,:]|=c
spec=np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(f)))); spec=np.clip((spec-np.percentile(spec,5))/(np.percentile(spec,99.9)-np.percentile(spec,5)),0,1)
fig,ax=plt.subplots(2,3,figsize=(10,6.8))
ims=[(f,'(a) Input image $f(x,y)$'),(spec,'(b) $\\log(1+|F(u,v)|)$'),
     (np.abs(g_ideal),'(c) Ideal high-pass, $D_0=0.05$'),(np.abs(g_gauss),'(d) Gaussian high-pass, $D_0=0.05$'),
     (g_log,'(e) LoG band-pass, $\\sigma=2$'),(~zc,'(f) Zero crossings of (e)')]
for a,(im,t) in zip(ax.ravel(),ims):
    if t.startswith('(c)') or t.startswith('(d)'): im=np.clip(im/np.percentile(im,99.5),0,1)
    a.imshow(im,cmap='gray'); a.set_title(t,fontsize=10); a.axis('off')
plt.tight_layout(); plt.savefig('fig_edges.pdf'); plt.close()

# zoom showing ringing
fig,ax=plt.subplots(1,2,figsize=(6.4,3.2))
for a,(im,t) in zip(ax,[(np.abs(g_ideal),'Ideal high-pass (ringing)'),(np.abs(g_gauss),'Gaussian high-pass')]):
    z=im[20:140,150:320]; a.imshow(np.clip(z/np.percentile(im,99.5),0,1),cmap='gray'); a.set_title(t,fontsize=10); a.axis('off')
plt.tight_layout(); plt.savefig('fig_ringing.pdf'); plt.close()

# filter profiles
r=np.linspace(0,0.5,500)
plt.figure(figsize=(6.4,2.6))
plt.plot(r,(r>D0).astype(float),label='Ideal high-pass')
plt.plot(r,1-np.exp(-r**2/(2*D0**2)),label='Gaussian high-pass')
hl=4*np.pi**2*r**2*np.exp(-2*np.pi**2*sigma**2*r**2); plt.plot(r,hl/hl.max(),label='$|H_{LoG}|$ (normalised)')
plt.plot(r,2*np.pi*r/(2*np.pi*0.5),'--',label='$|j2\\pi\\rho|$ (normalised)')
rs=1/(np.sqrt(2)*np.pi*sigma); plt.axvline(rs,color='gray',lw=0.8,ls=':'); plt.text(rs+0.005,0.05,'$\\rho^*=1/(\\sqrt{2}\\pi\\sigma)$',fontsize=9)
plt.xlabel('radial frequency $\\rho$ (cycles/pixel)'); plt.ylabel('$|H(\\rho)|$'); plt.legend(fontsize=8,loc='center right'); plt.tight_layout()
plt.savefig('fig_filters.pdf'); plt.close()

# ---------- Figure 2: texture segmentation with Gabor energy ----------
rng=np.random.default_rng(1); S=256
y,x=np.mgrid[0:S,0:S]
tA=np.sin(2*np.pi*(0.10*x))                       # vertical stripes, 0.10 cyc/px
tB=np.sin(2*np.pi*(0.06*(x*np.cos(np.pi/3)+y*np.sin(np.pi/3))))  # oblique, 0.06 cyc/px
region=((x-150)**2/80**2+(y-120)**2/60**2)<1
img=np.where(region,tB,tA)+0.6*rng.standard_normal((S,S))
u,v,D=freq_grid(S,S)
def gabor_H(u0,v0,sg): return np.exp(-2*np.pi**2*sg**2*((u-u0)**2+(v-v0)**2))
sg=6
HA=gabor_H(0.10,0,sg)+gabor_H(-0.10,0,sg)            # tuned to texture A (u axis = columns)
# careful: fft2 axis0=rows(y), axis1=cols(x); freq_grid u is along rows. Build explicitly:
fy=np.fft.fftfreq(S)[:,None]; fx=np.fft.fftfreq(S)[None,:]
def gH(fx0,fy0): return np.exp(-2*np.pi**2*sg**2*((fx-fx0)**2+(fy-fy0)**2))
HA=gH(0.10,0)+gH(-0.10,0)
bx,by=0.06*np.cos(np.pi/3),0.06*np.sin(np.pi/3)
HB=gH(bx,by)+gH(-bx,-by)
F=np.fft.fft2(img)
EA=np.abs(np.fft.ifft2(F*HA))**2; EB=np.abs(np.fft.ifft2(F*HB))**2
# local energy: smooth with Gaussian (itself a frequency-domain low-pass)
Lp=np.exp(-2*np.pi**2*8**2*(fx**2+fy**2))
EA=np.real(np.fft.ifft2(np.fft.fft2(EA)*Lp)); EB=np.real(np.fft.ifft2(np.fft.fft2(EB)*Lp))
Rt=np.log(EB+1e-9)-np.log(EA+1e-9)
seg=(Rt>0)
acc=(seg==region).mean()
spec=np.log1p(np.abs(np.fft.fftshift(F)))
fig,ax=plt.subplots(1,4,figsize=(12,3.3))
ax[0].imshow(img,cmap='gray'); ax[0].set_title('(a) Two textures + noise',fontsize=10)
ax[1].imshow(spec,cmap='gray',extent=[-.5,.5,.5,-.5]); ax[1].set_title('(b) $\\log(1+|F|)$: two peak pairs',fontsize=10); ax[1].set_xlabel('$u$'); ax[1].set_ylabel('$v$')
ax[2].imshow(Rt,cmap='coolwarm'); ax[2].set_title('(c) $\\log E_B-\\log E_A$',fontsize=10)
ax[3].imshow(img,cmap='gray'); ax[3].contour(seg,[0.5],colors='r',linewidths=1.5); ax[3].contour(region,[0.5],colors='lime',linewidths=1,linestyles='--')
ax[3].set_title('(d) Boundary (red) vs truth (green)',fontsize=10)
for a in [ax[0],ax[2],ax[3]]: a.axis('off')
plt.tight_layout(); plt.savefig('fig_texture.pdf'); plt.close()
print('texture pixel accuracy',round(acc,4))
