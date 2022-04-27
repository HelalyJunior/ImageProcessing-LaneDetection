# import all components
# from the tkinter library
from tkinter import *

# import filedialog module
from tkinter import filedialog


# Function for opening the
# file explorer window
def browseFiles():
	filename = filedialog.askopenfilename(initialdir = "/",
										title = "Select a File",
										filetypes = (("Text files",
														"*.txt*"),
													("all files",
														"*.*")))
	
	# Change label contents
	label_file_explorer.configure(text="File Opened: "+filename)
 
 
def browseDirectory():
    
    direcName = filedialog.askdirectory(initialdir = "/",
                                    title = "Select a directory",
                                   )

    # Change label contents
    dir_file_explorer.configure(text="Directory Chosen: "+direcName)
    


def click_me():
    print(i.get())
    t = label_file_explorer.cget("text")[13:]
    j= dir_file_explorer.cget("text")[18:]
    with open("run.bat","w+") as f :
        f.write("call conda activate MLenv\n")
        f.write(f"py curvedTrying.py {t} {i.get()} {j}/"+'\n')
        f.write("call conda deactivate")
        
        
    
	
	
																								
# Create the root window
window = Tk()

# Set window title
window.title('File Explorer')

# Set window size
window.geometry("500x500")

#Set window background color
window.config(background = "white")
i= IntVar()
c = Checkbutton(window, text = "Debug Mode" , variable=i)

b = Button(window,text="Launch Code ! ",command=click_me)


# Create a File Explorer label
label_file_explorer = Label(window,
							text = "File Explorer using Tkinter",
							width = 100, height = 4,
							fg = "blue")

# Create a Directory Explorer label
dir_file_explorer = Label(window,
							text = "directory Explorer using Tkinter",
							width = 100, height = 4,
							fg = "blue")

button_dir = Button(window,
						text = "Choose Destination Dir",
						command = browseDirectory)

	
button_explore = Button(window,
						text = "Choose Source Video",
						command = browseFiles)


button_exit = Button(window,
					text = "Exit",
					command = exit)

# Grid method is chosen for placing
# the widgets at respective positions
# in a table like structure by
# specifying rows and columns
label_file_explorer.grid(column = 1, row = 1)

c.grid(column=1,row=4)
b.grid(column=1,row=5)
button_dir.grid(column=1,row=6)
dir_file_explorer.grid(column=1,row=7)




button_explore.grid(column = 1, row = 2)

button_exit.grid(column = 1,row = 8)

# Let the window wait for any events
window.mainloop()
