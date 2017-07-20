(ns learn-cortex.core
  (:require [cortex.experiment.train :as train]
            [cortex.nn.execute :as execute]
            [cortex.nn.layers :as layers]
            [cortex.nn.network :as network]))

(def xor
  [{:x [0.0 0.0] :y [0.0]}
   {:x [0.0 1.0] :y [1.0]}
   {:x [1.0 0.0] :y [1.0]}
   {:x [1.0 1.0] :y [0.0]}])

(def nn (-> [(layers/input 2 1 1 :id :x) ;; input :x 2*1 dimensions
             (layers/linear->tanh 10)
             (layers/linear 1 :id :y)]
            network/linear-network))

(def trained
  ;; Train with xor, test with xor
  (train/train-n nn xor xor :batch-size 4 :epoch-count 3000))

;; Before
(execute/run nn xor)

;; After
(execute/run trained xor)
