from gui import App
import logdecoder


"""
Parameters of the program run can be set here:

log_id: the id that is added to the log file for identification
mode: sets the type of conversation. In both scenarios the user is presented a target and a distractor color. Possible values are
        'learn': The target object is indicated with an arrow and the user is asked to give a description.
        'validate': No object is marked. The user is given a description and is asked to pick which object he or she thinks is the target.
learn_mode: determines which learning algorithm is used in 'learn' mode. Has no effect if mode is 'validate'. Possible values are 
        'statistical', 'naive', 'context-sensitive'
"""

# ---------------------- #
log_id = 0
mode = 'learn'
learn_mode = 'naive'
# ---------------------- #



open_log_mode = 'a'
if mode == 'learn':
    open_log_mode = 'w'

if __name__ == '__main__':
    app = App(log_id, open_log_mode)

    if not app.set_learn_mode(learn_mode + '!'):
        raise Exception("Invalid Learn Mode!")

    if mode == 'learn':
        app.new_learn()
    elif mode == 'validate':
        validation_entries = logdecoder.decode_log(log_id)
        app.new_validation(validation_entries)
    
    app.mainloop()

