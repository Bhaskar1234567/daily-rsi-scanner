import tkinter as tk
from tkinter import ttk, messagebox
import threading, time, json
from urllib.request import Request, urlopen
from datetime import datetime, timezone, timedelta

BASE="https://fapi.binance.com"
IST=timezone(timedelta(hours=5,minutes=30))
COINS=["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","AVAXUSDT","TRXUSDT","LINKUSDT","DOTUSDT","LTCUSDT","BCHUSDT","UNIUSDT","ATOMUSDT","ETCUSDT","FILUSDT","NEARUSDT","APTUSDT","ARBUSDT","OPUSDT","INJUSDT","SUIUSDT","SEIUSDT","TIAUSDT","AAVEUSDT","SHIBUSDT","PEPEUSDT","FLOKIUSDT","WIFUSDT"]

def api(path, params):
    q="&".join(f"{k}={v}" for k,v in params.items())
    req=Request(BASE+path+"?"+q,headers={"User-Agent":"BoxScanner/1.0"})
    with urlopen(req,timeout=10) as r:return json.loads(r.read())

def get_data(sym):
    now=int(time.time()*1000)
    d=api("/fapi/v1/klines",{"symbol":sym,"interval":"1d","limit":3})
    prev=[x for x in d if x[6]<now][-1]
    o,h,l,c=map(float,[prev[1],prev[2],prev[3],prev[4]])
    hi=max((h+c)/2,(o+l)/2); lo=min((h+c)/2,(o+l)/2)
    utc=datetime.now(timezone.utc); start=int(datetime(utc.year,utc.month,utc.day,tzinfo=timezone.utc).timestamp()*1000)
    k=api("/fapi/v1/klines",{"symbol":sym,"interval":"3m","startTime":start,"limit":2})
    last=k[-1] if k[-1][6]<now else k[-2]
    first=api("/fapi/v1/klines",{"symbol":sym,"interval":"3m","startTime":start,"limit":1})[0]
    price=float(last[4]); vol=float(last[5]); op=float(first[1])
    pos="BELOW BOX" if price<lo else "ABOVE BOX" if price>hi else "IN BOX"
    qualified=lo<=op<=hi
    signal="UP BREAKOUT" if qualified and vol>=MINVOL*VMULT and price>hi else "DOWN BREAKOUT" if qualified and vol>=MINVOL*VMULT and price<lo else "IN BOX / WAITING" if qualified else "NOT QUALIFIED"
    return dict(coin=sym,price=price,volume=vol,dayopen=op,low=lo,high=hi,pos=pos,signal=signal,time=int(last[0]))

MINVOL=1000000.0
VMULT=1.0

class App:
    def __init__(self,root):
        self.root=root; self.running=False; self.states={}; self.breaks=[]
        root.title("Previous-Day Box 3M Dashboard Scanner"); root.geometry("1450x800")
        bar=ttk.Frame(root,padding=8); bar.pack(fill="x")
        ttk.Label(bar,text="PREVIOUS-DAY BOX • 3M CRYPTO SCANNER",font=("Segoe UI",16,"bold")).pack(side="left")
        ttk.Label(bar,text=" Min Volume:").pack(side="left",padx=(20,3))
        self.mv=tk.StringVar(value="1000000"); ttk.Entry(bar,textvariable=self.mv,width=10).pack(side="left")
        ttk.Label(bar,text=" Volume ×:").pack(side="left",padx=(10,3))
        self.mx=tk.StringVar(value="1.0"); ttk.Entry(bar,textvariable=self.mx,width=6).pack(side="left")
        self.rf=tk.StringVar(value="15"); ttk.Label(bar,text=" Refresh:").pack(side="left",padx=(10,3)); ttk.Entry(bar,textvariable=self.rf,width=5).pack(side="left")
        self.startb=ttk.Button(bar,text="START 24/7",command=self.start); self.startb.pack(side="left",padx=8)
        self.stopb=ttk.Button(bar,text="STOP",command=self.stop,state="disabled"); self.stopb.pack(side="left")
        self.status=tk.StringVar(value="Stopped"); ttk.Label(bar,textvariable=self.status).pack(side="right")
        ttk.Label(root,text="IN BOX / WAITING",font=("Segoe UI",12,"bold")).pack(anchor="w",padx=10)
        f=ttk.Frame(root,padding=8); f.pack(fill="both",expand=True)
        cols=("coin","price","open","low","high","pos","vol","signal"); self.wait=ttk.Treeview(f,columns=cols,show="headings",height=12)
        heads=["COIN","PRESENT PRICE","TODAY OPEN","LOWER BREAKOUT","UPPER BREAKOUT","POSITION","3M VOLUME","STATUS"]
        for c,h in zip(cols,heads):self.wait.heading(c,text=h);self.wait.column(c,width=150,anchor="center")
        self.wait.pack(fill="both",expand=True)
        ttk.Label(root,text="CONFIRMED BREAKOUTS • NEWEST FIRST",font=("Segoe UI",12,"bold")).pack(anchor="w",padx=10,pady=6)
        b=ttk.Frame(root,padding=8); b.pack(fill="both",expand=True)
        cols2=("n","coin","time","direction","bp","price","low","high","vol"); self.bt=ttk.Treeview(b,columns=cols2,show="headings",height=8)
        heads2=["#","COIN","BREAKOUT CANDLE","CONFIRMATION","BREAK PRICE","PRESENT PRICE","LOWER LEVEL","UPPER LEVEL","3M VOLUME"]
        for c,h in zip(cols2,heads2):self.bt.heading(c,text=h);self.bt.column(c,width=140,anchor="center")
        self.bt.pack(fill="both",expand=True)
        root.protocol("WM_DELETE_WINDOW",self.close)

    def start(self):
        global MINVOL,VMULT
        try: MINVOL=float(self.mv.get()); VMULT=float(self.mx.get()); r=max(5,int(self.rf.get()))
        except: messagebox.showerror("Error","Enter valid settings."); return
        self.running=True; self.startb.config(state="disabled"); self.stopb.config(state="normal"); self.status.set("RUNNING")
        threading.Thread(target=self.loop,args=(r,),daemon=True).start()

    def stop(self):
        self.running=False; self.startb.config(state="normal"); self.stopb.config(state="disabled"); self.status.set("Stopped")

    def loop(self,r):
        while self.running:
            for s in COINS:
                if not self.running: break
                try:
                    d=get_data(s)
                    old=self.states.get(s)
                    self.states[s]=d
                    if d["signal"] in ("UP BREAKOUT","DOWN BREAKOUT") and (not old or old.get("signal") not in ("UP BREAKOUT","DOWN BREAKOUT")):
                        self.breaks.append(d)
                except Exception as e:
                    self.states[s]={"coin":s,"signal":"ERROR"}
            self.root.after(0,self.refresh)
            for _ in range(r):
                if not self.running:break
                time.sleep(1)

    def refresh(self):
        for x in self.wait.get_children():self.wait.delete(x)
        for s in COINS:
            d=self.states.get(s)
            if d and d.get("signal")=="IN BOX / WAITING":
                self.wait.insert("", "end", values=(s,fmt(d["price"]),fmt(d["dayopen"]),fmt(d["low"]),fmt(d["high"]),d["pos"],vol(d["volume"]),d["signal"]))
        for x in self.bt.get_children():self.bt.delete(x)
        bs=sorted(self.breaks,key=lambda x:x["time"],reverse=True)
        for i,d in enumerate(bs,1):
            tm=datetime.fromtimestamp(d["time"]/1000,timezone.utc).astimezone(IST).strftime("%Y-%m-%d %H:%M:%S IST")
            self.bt.insert("", "end", values=(i,d["coin"],tm,d["signal"],fmt(d["price"]),fmt(d["price"]),fmt(d["low"]),fmt(d["high"]),vol(d["volume"])))

    def close(self):self.running=False;self.root.destroy()

def fmt(x):
    if x is None:return "-"
    return f"{x:,.8f}" if abs(x)<1 else f"{x:,.4f}"
def vol(x):
    return f"{x/1e9:.2f}B" if x>=1e9 else f"{x/1e6:.2f}M" if x>=1e6 else f"{x/1e3:.2f}K"

if __name__=="__main__":
    App(tk.Tk()).root.mainloop()
